﻿#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <ompl/datastructures/BinaryHeap.h>
#include <ompl/tools/config/SelfConfig.h>

#include <ompl/datastructures/NearestNeighborsGNAT.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/fmt/BFMT.h>

#include <fstream>
#include <ompl/base/spaces/RealVectorStateSpace.h>

// unvisited set: nodes that not been added to the trees yet
// open set: vertices that have been added to the trees, and at the front line for expansion
// close set: vertices that have been added to the trees and would never considered for expansion
namespace ompl
{
   namespace geometric
   {
       BFMT::BFMT(const base::SpaceInformationPtr &si)
         : base::Planner(si, "BFMT")
         , freeSpaceVolume_(si_->getStateSpace()->getMeasure())  // An upper bound on the free space volume is the
                                                                 // total space volume; the free fraction is estimated
                                                                 // in sampleFree
       {
           connect_motion_=nullptr;
           specs_.approximateSolutions = false;
           specs_.directed = false;

           ompl::base::Planner::declareParam<unsigned int>("num_samples", this, &BFMT::setNumSamples,
                                                           &BFMT::getNumSamples, "10:10:1000000");
           ompl::base::Planner::declareParam<double>("radius_multiplier", this, &BFMT::setRadiusMultiplier,
                                                     &BFMT::getRadiusMultiplier, "0.1:0.05:50.");
           ompl::base::Planner::declareParam<bool>("nearest_k", this, &BFMT::setNearestK, &BFMT::getNearestK, "0,1");
           ompl::base::Planner::declareParam<bool>("balanced", this, &BFMT::setExploration, &BFMT::getExploration,
                                                   "0,1");
           ompl::base::Planner::declareParam<bool>("optimality", this, &BFMT::setTermination, &BFMT::getTermination,
                                                   "0,1");
           ompl::base::Planner::declareParam<bool>("heuristics", this, &BFMT::setHeuristics, &BFMT::getHeuristics,
                                                   "0,1");
           ompl::base::Planner::declareParam<bool>("cache_cc", this, &BFMT::setCacheCC, &BFMT::getCacheCC, "0,1");
           ompl::base::Planner::declareParam<bool>("extended_fmt", this, &BFMT::setExtendedFMT, &BFMT::getExtendedFMT,
                                                   "0,1");
           ompl::base::Planner::declareParam<double>("batchfactor", this, &BFMT::setBatchFactor, &BFMT::getBatchFactor, "0.1:0.05:50.");
            ompl::base::Planner::declareParam<unsigned int>("terminatetime", this, &BFMT::setTerminatetime, &BFMT::getTerminatetime, "1:1:500");
           addPlannerProgressProperty("best cost REAL", [this] { return std::to_string(lastCost().value()); });
           addPlannerProgressProperty("iterations INTEGER", [this] { return std::to_string(numIterations()); });
           addPlannerProgressProperty("sample count INTEGER", [this]
                                      {
                                          return std::to_string(numSamples());
                                      });
           addPlannerProgressProperty("collision checks INTEGER", [this]
                                      {
                                          return std::to_string(numCollisionChecks());
                                      });
           addPlannerProgressProperty("extendFMT times INTEGER", [this]
                                      {
                                          return std::to_string(numExtend());
                                      });//*
           addPlannerProgressProperty("iteration times INTEGER", [this]
                                      {
                                          return std::to_string(numIterTimes());
                                      });
           addPlannerProgressProperty("validsample INTEGER", [this]
                                      {
                                          return std::to_string(validsample());
                                      });
           addPlannerProgressProperty("iteration cost REAL", [this]
                                      {
                                          return std::to_string(iterationCost().value());
                                      });
       }

       ompl::geometric::BFMT::~BFMT()
       {
           freeMemory();
       }

       void BFMT::setup()
       {
           if (pdef_)
           {
               /* Setup the optimization objective. If no optimization objective was
               specified, then default to optimizing path length as computed by the
               distance() function in the state space */
               if (pdef_->hasOptimizationObjective())
                   opt_ = pdef_->getOptimizationObjective();
               else
               {
                   OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length.",
                               getName().c_str());
                   opt_ = std::make_shared<base::PathLengthOptimizationObjective>(si_);
                   // Store the new objective in the problem def'n
                   pdef_->setOptimizationObjective(opt_);
               }
               Open_[0].getComparisonOperator().opt_ = opt_.get();
               Open_[0].getComparisonOperator().heuristics_ = heuristics_;
               Open_[1].getComparisonOperator().opt_ = opt_.get();
               Open_[1].getComparisonOperator().heuristics_ = heuristics_;

               if (!nn_)
                   nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<BiDirMotion *>(this));
               nn_->setDistanceFunction([this](const BiDirMotion *a, const BiDirMotion *b)
                                        {
                                            return distanceFunction(a, b);
                                        });

               if (nearestK_ && !nn_->reportsSortedResults())
               {
                   OMPL_WARN("%s: NearestNeighbors datastructure does not return sorted solutions. Nearest K strategy "
                             "disabled.",
                             getName().c_str());
                   nearestK_ = false;
               }
           }
           else
           {
               OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
               setup_ = false;
           }
       }

       void BFMT::freeMemory()
       {
           if (nn_)
           {
               BiDirMotionPtrs motions;
               nn_->list(motions);
               for (auto &motion : motions)
               {
                   si_->freeState(motion->getState());
                   delete motion;
               }
           }
       }

       void BFMT::clear()
       {
           Planner::clear();

           connect_motion_=nullptr;//modify
//           setNumSamples(numSamples_);
           sampler_.reset();
           freeMemory();
           if (nn_)
               nn_->clear();
           Open_[FWD].clear();
           Open_[REV].clear();
           Open_elements[FWD].clear();
           Open_elements[REV].clear();
           neighborhoods_.clear();
           lastCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
           iterationcost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
           collisionChecks_ = 0;
           iterations_ = 0;
           sampleCount_ = 0;
           extendCount_ = 0;
           validsample_ = 0;
           iterTimes_ = 0;
       }

       void BFMT::getPlannerData(base::PlannerData &data) const
       {
           base::Planner::getPlannerData(data);
           BiDirMotionPtrs motions;
           nn_->list(motions);

           int numStartNodes = 0;
           int numGoalNodes = 0;
           int numEdges = 0;
           int numFwdEdges = 0;
           int numRevEdges = 0;

           int fwd_tree_tag = 1;
           int rev_tree_tag = 2;

           for (auto motion : motions)
           {
               bool inFwdTree = (motion->currentSet_[FWD] != BiDirMotion::SET_UNVISITED);

               // For samples added to the fwd tree, add incoming edges (from fwd tree parent)
               if (inFwdTree)
               {
                   if (motion->parent_[FWD] == nullptr)
                   {
                       // Motion is a forward tree root node
                       ++numStartNodes;
                   }
                   else
                   {
                       bool success =
                           data.addEdge(base::PlannerDataVertex(motion->parent_[FWD]->getState(), fwd_tree_tag),
                                        base::PlannerDataVertex(motion->getState(), fwd_tree_tag));
                       if (success)
                       {
                           ++numFwdEdges;
                           ++numEdges;
                       }
                   }
               }
           }

           // The edges in the goal tree are reversed so that they are in the same direction as start tree
           for (auto motion : motions)
           {
               bool inRevTree = (motion->currentSet_[REV] != BiDirMotion::SET_UNVISITED);

               // For samples added to a tree, add incoming edges (from fwd tree parent)
               if (inRevTree)
               {
                   if (motion->parent_[REV] == nullptr)
                   {
                       // Motion is a reverse tree root node
                       ++numGoalNodes;
                   }
                   else
                   {
                       bool success =
                           data.addEdge(base::PlannerDataVertex(motion->getState(), rev_tree_tag),
                                        base::PlannerDataVertex(motion->parent_[REV]->getState(), rev_tree_tag));
                       if (success)
                       {
                           ++numRevEdges;
                           ++numEdges;
                       }
                   }
               }
           }
       }

       void BFMT::saveNeighborhood(const std::shared_ptr<NearestNeighbors<BiDirMotion *>> &nn, BiDirMotion *m)
       {
           // Check if neighborhood has already been saved
           if (neighborhoods_.find(m) == neighborhoods_.end())
           {
               BiDirMotionPtrs neighborhood;
               if (nearestK_)
                   nn_->nearestK(m, NNk_, neighborhood);
               else
                   nn_->nearestR(m, NNr_, neighborhood);

               if (!neighborhood.empty())
               {
                   // Save the neighborhood but skip the first element (m)
                   neighborhoods_[m] = std::vector<BiDirMotion *>(neighborhood.size() - 1, nullptr);
                   std::copy(neighborhood.begin() + 1, neighborhood.end(), neighborhoods_[m].begin());
               }
               else
               {
                   // Save an empty neighborhood
                   neighborhoods_[m] = std::vector<BiDirMotion *>(0);
               }
           }
       }

       void BFMT::sampleFree(const std::shared_ptr<NearestNeighbors<BiDirMotion *>> &nn,
                             const base::PlannerTerminationCondition &ptc)
       {
           if(connect_motion_==nullptr)
           {
//           std::cout << "sample begin" << std::endl;//test code
           unsigned int nodeCount = 0;
           unsigned int sampleAttempts = 0;
           auto *motion = new BiDirMotion(si_, &tree_);

           // Sample numSamples_ number of nodes from the free configuration space
           while (nodeCount < numSamples_ && !ptc)
           {
               sampler_->sampleUniform(motion->getState());
               ++sampleCount_;
               sampleAttempts++;
               if (si_->isValid(motion->getState()))
               {  // collision checking
                   ++nodeCount;
                   nn->add(motion);
                   motion = new BiDirMotion(si_, &tree_);
                   ++validsample_;
               }
           }
           si_->freeState(motion->getState());
           delete motion;

           // 95% confidence limit for an upper bound for the true free space volume
           freeSpaceVolume_ =
               boost::math::binomial_distribution<>::find_upper_bound_on_p(sampleAttempts, nodeCount, 0.05) *
               si_->getStateSpace()->getMeasure();
//           std::cout << "sample end" << std::endl;//test code
           }
           else
           {
               base::InformedSamplerPtr infSampler_; // infomed采样器，单点采样
               infSampler_.reset();  // reset informed sampler
               if (static_cast<bool>(opt_) == true)
               {
                   if (opt_->hasCostToGoHeuristic() == false)
                   {
                       OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
                   }
               }
               // We are using informed sampling, this can end-up reverting to rejection sampling in some cases
               OMPL_INFORM("%s: Using informed sampling.", getName().c_str());
               infSampler_ = opt_->allocInformedStateSampler(pdef_, 100u);
               unsigned int nodeCount = 0;
               unsigned int sampleAttempts = 0;
               auto *motion = new BiDirMotion(si_, &tree_);

               // Sample numSamples_ number of nodes from the free configuration space
               while (nodeCount < batchnumSamples_ && !ptc)
               {
                   infSampler_->sampleUniform(motion->getState(), lastCost_);
                   ++sampleCount_;
                   sampleAttempts++;

                   bool collision_free = si_->isValid(motion->getState());

                   if (collision_free)
                   {
                       nodeCount++;
                       nn->add(motion);
                       motion = new BiDirMotion(si_, &tree_);
                       ++validsample_;
                   }  // If collision free
               }      // While nodeCount < numSamples
               si_->freeState(motion->getState());
               delete motion;

               // 95% confidence limit for an upper bound for the true free space volume
               freeSpaceVolume_ = boost::math::binomial_distribution<>::find_upper_bound_on_p(sampleAttempts, nodeCount, 0.05) *
                       si_->getStateSpace()->getMeasure();
           }
        }

       double BFMT::calculateUnitBallVolume(const unsigned int dimension) const
       {
           if (dimension == 0)
               return 1.0;
           if (dimension == 1)
               return 2.0;
           return 2.0 * boost::math::constants::pi<double>() / dimension * calculateUnitBallVolume(dimension - 2);
       }

       double BFMT::calculateRadius(const unsigned int dimension, const unsigned int n) const
       {
           double a = 1.0 / (double)dimension;
           double unitBallVolume = calculateUnitBallVolume(dimension);

           return radiusMultiplier_ * 2.0 * std::pow(a, a) * std::pow(freeSpaceVolume_ / unitBallVolume, a) *
                  std::pow(log((double)n) / (double)n, a);
       }

       void BFMT::initializeProblem(base::GoalSampleableRegion *&goal_s)
       {
           checkValidity();
           if (!sampler_)
           {
               sampler_ = si_->allocStateSampler();
           }
           goal_s = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());
       }

       base::PlannerStatus BFMT::solve(const base::PlannerTerminationCondition &ptc)
       {
           base::GoalSampleableRegion *goal_s;
           initializeProblem(goal_s);
           if (goal_s == nullptr)
           {
               OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
               return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
           }

           useFwdTree();

           // Add start states to Unvisitedfwd and Openfwd
           bool valid_initMotion = false;
            BiDirMotion *initMotion;
           while (const base::State *st = pis_.nextStart())
           {
               initMotion = new BiDirMotion(si_, &tree_);
               si_->copyState(initMotion->getState(), st);

               initMotion->currentSet_[REV] = BiDirMotion::SET_UNVISITED;
               nn_->add(initMotion);  // S <-- {x_init}
               if (si_->isValid(initMotion->getState()))
               {
                   // Take the first valid initial state as the forward tree root
                   Open_elements[FWD][initMotion] = Open_[FWD].insert(initMotion);
                   initMotion->currentSet_[FWD] = BiDirMotion::SET_OPEN;
                   initMotion->cost_[FWD] = opt_->initialCost(initMotion->getState());
                   valid_initMotion = true;
                   heurGoalState_[1] = initMotion->getState();
               }
           }

           if ((initMotion == nullptr) || !valid_initMotion)
           {
               OMPL_ERROR("Start state undefined or invalid.");
               return base::PlannerStatus::INVALID_START;
           }

            // Sample N free states in configuration state_
           if (!sampler_)
               sampler_ = si_->allocStateSampler(); // 开辟采样内存
           else
           {
               sampler_.reset();
               sampler_ = si_->allocStateSampler();
           }
          setNumSamples(numSamples_);
//          std::cout<<"numSamples_: "<< numSamples_<<std::endl;
            sampleFree(nn_, ptc);  // S <-- SAMPLEFREE(N)
            OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(),
                        nn_->size());

            // Calculate the nearest neighbor search radius
            if (nearestK_)
            {
                NNk_ = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                                 (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                                 log((double)nn_->size()));
                OMPL_DEBUG("Using nearest-neighbors k of %d", NNk_);
            }
            else
            {
                NNr_ = calculateRadius(si_->getStateDimension(), nn_->size());
                OMPL_DEBUG("Using radius of %f", NNr_);
            }

           // Add goal states to Unvisitedrev and Openrev
           bool valid_goalMotion = false;
            BiDirMotion *goalMotion;
           while (const base::State *st = pis_.nextGoal())
           {
               goalMotion = new BiDirMotion(si_, &tree_);
               si_->copyState(goalMotion->getState(), st);

//               goalState_=goalMotion->getState();//modify

               goalMotion->currentSet_[FWD] = BiDirMotion::SET_UNVISITED;
               nn_->add(goalMotion);  // S <-- {x_goal}
               if (si_->isValid(goalMotion->getState()))
               {
                   // Take the first valid goal state as the reverse tree root
                   Open_elements[REV][goalMotion] = Open_[REV].insert(goalMotion);
                   goalMotion->currentSet_[REV] = BiDirMotion::SET_OPEN;
                   goalMotion->cost_[REV] = opt_->terminalCost(goalMotion->getState());
                   valid_goalMotion = true;
                   heurGoalState_[0] = goalMotion->getState();
               }
           }

           if ((goalMotion == nullptr) || !valid_goalMotion)
           {
               OMPL_ERROR("Goal state undefined or invalid.");
               return base::PlannerStatus::INVALID_GOAL;
           }

           useRevTree();

           // Plan a path
           BiDirMotion *connection_point = nullptr;
           bool earlyFailure = true;

           if (initMotion != nullptr && goalMotion != nullptr)
           {
               earlyFailure = plan(initMotion, goalMotion, connection_point, ptc);
           }
           else
           {
               OMPL_ERROR("Initial/goal state(s) are undefined!");
           }

           if (earlyFailure)
           {
               return base::PlannerStatus(false, false);
           }


           // Save the best path (through z)
           if (!ptc)
           {
               base::Cost fwd_cost, rev_cost, connection_cost;

               // Construct the solution path
               useFwdTree();
               BiDirMotionPtrs path_fwd;
               tracePath(connection_point, path_fwd);
               fwd_cost = connection_point->getCost();

               useRevTree();
               BiDirMotionPtrs path_rev;
               tracePath(connection_point, path_rev);
               rev_cost = connection_point->getCost();
               // ASSUMES FROM THIS POINT THAT z = path_fwd[0] = path_rev[0]
               // Remove the first element, z, in the traced reverse path
               // (the same as the first element in the traced forward path)
               if (path_rev.size() > 1)
               {
                   connection_cost = base::Cost(rev_cost.value() - path_rev[1]->getCost().value());
                   path_rev.erase(path_rev.begin());
               }
               else if (path_fwd.size() > 1)
               {
                   connection_cost = base::Cost(fwd_cost.value() - path_fwd[1]->getCost().value());
                   path_fwd.erase(path_fwd.begin());
               }
               else
               {
                   OMPL_ERROR("Solution path traced incorrectly or otherwise constructed improperly \
               through forward/reverse trees (both paths are one node in length, each).");
               }

               // Adjust costs/parents in reverse tree nodes as cost/direction from forward tree root
               useFwdTree();
               path_rev[0]->setCost(base::Cost(path_fwd[0]->getCost().value() + connection_cost.value()));
               path_rev[0]->setParent(path_fwd[0]);
               for (unsigned int i = 1; i < path_rev.size(); ++i)
               {
                   path_rev[i]->setCost(
                       base::Cost(fwd_cost.value() + (rev_cost.value() - path_rev[i]->getCost().value())));
                   path_rev[i]->setParent(path_rev[i - 1]);
               }

               BiDirMotionPtrs mpath_;
               mpath_.clear();
               std::reverse(path_rev.begin(), path_rev.end());
               mpath_.reserve(path_fwd.size() + path_rev.size());  // preallocate memory
               mpath_.insert(mpath_.end(), path_rev.begin(), path_rev.end());
               mpath_.insert(mpath_.end(), path_fwd.begin(), path_fwd.end());


               // Set the solution path
               auto path(std::make_shared<PathGeometric>(si_));
               for (int i = mpath_.size() - 1; i >= 0; --i)
               {
                   path->append(mpath_[i]->getState());
               }
//               mPathSize_ = mpath_.size();
                OMPL_DEBUG("total path cost: %f\n", lastCost_.value());
               static const bool approximate = false;
               static const double cost_difference_from_goal = 0.0;
               pdef_->addSolutionPath(path, approximate, cost_difference_from_goal, getName());

//               drawPathe();
                OMPL_DEBUG("Total path cost: %f\n", fwd_cost.value() + rev_cost.value());
               return base::PlannerStatus(true, false);
           }

           // Planner terminated without accomplishing goal
           return base::PlannerStatus(false, false);
       }

       void BFMT::traceSolutionPathThroughTree(BiDirMotion *&connection_point)
       {
           base::Cost fwd_cost, rev_cost, connection_cost;
           BiDirMotionPtrs pwdpath_;
           BiDirMotionPtrs revpath_;
           // Construct the solution path
           if(tree_==FWD)
           {

//           BiDirMotionPtrs path_fwd;
           tracePath(connection_point, pwdpath_);
           fwd_cost = connection_point->getCost();
//           pwdPathSize_ = pwdpath_.size();

           swapTrees();

//           BiDirMotionPtrs path_rev;
           tracePath(connection_point, revpath_);
           rev_cost = connection_point->getCost();
//           revPathSize_ = revpath_.size();
           swapTrees();
           }
           else
           {
               revpath_.clear();
//               BiDirMotionPtrs path_rev;
               tracePath(connection_point, revpath_);
               rev_cost = connection_point->getCost();
//               revPathSize_ = revpath_.size();

               swapTrees();
//               BiDirMotionPtrs path_fwd;
               pwdpath_.clear();
               tracePath(connection_point, pwdpath_);
               fwd_cost = connection_point->getCost();
//               pwdPathSize_ = pwdpath_.size();
               swapTrees();
           }
           lastCost_ = opt_->combineCosts(fwd_cost,rev_cost);//modify

//           std::reverse(path_rev.begin(), path_rev.end());
////            if(path_rev[0]->getParent()!=nullptr)
////                std::cout<<"path_rev[0]是connect_point"<< std::endl;
////            else
////                std::cout<<"path_rev[0]不是connect_point"<<std::endl;
////            removeFromParent(path_rev[0]);
//           std::vector<BiDirMotion*>::iterator iter = std::find(path_rev[0]->getParent()->getChildren().begin()
//                                                           ,path_rev[0]->getParent()->getChildren().end()
//                                                           ,path_rev[0]);
//           //vector< int >::iterator iter=std::find(v.begin(),v.end(),num_to_find); //返回的是一个迭代器指针
//           if(iter != path_rev[0]->getParent()->getChildren().end())//如果能找到hhNear，则在hhNear->getParent()->getChildren()里删除hhNear
//           {   if(path_rev[0]->getParent()!=nullptr)
//               path_rev[0]->getParent()->getChildren().erase(iter);
//           }

////            if(path_rev[0]->getParent()!=nullptr)
////                std::cout<<"path_rev[0]是connect_point"<< std::endl;
////            else
////                std::cout<<"path_rev[0]不是connect_point"<<std::endl;
//           for (unsigned int i = 1; i < path_rev.size(); ++i)
//           {
////                if(path_rev[i]->getParent()==path_rev[i-1])
////                    std::cout<<"path_rev[i]->getParent()==path_rev[i-1]"<<std::endl;
////                else
////                    std::cout<<"path_rev[i]->getParent()!=path_rev[i-1]"<<std::endl;
////                removeFromParent(path_rev[i]);
//               std::vector<BiDirMotion*>::iterator iter = std::find(path_rev[i]->getParent()->getChildren().begin()
//                                                               ,path_rev[i]->getParent()->getChildren().end()
//                                                               ,path_rev[i]);
//               //vector< int >::iterator iter=std::find(v.begin(),v.end(),num_to_find); //返回的是一个迭代器指针
//               if(iter != path_rev[i]->getParent()->getChildren().end())//如果能找到hhNear，则在hhNear->getParent()->getChildren()里删除hhNear
//               {
//                   std::cout<<"already to remove from parent"<< std::endl;
//                   if(path_rev[i]->getParent()!=nullptr)
//                       path_rev[i]->getParent()->getChildren().erase(iter);
////                    if(path_rev[i]->getParent()==path_rev[i-1])
////                        std::cout<<"path_rev[i]->getParent()==path_rev[i-1]"<<std::endl;
////                    else
////                        std::cout<<"path_rev[i]->getParent()!=path_rev[i-1]"<<std::endl;
//               }
//           }
       }
// expansion function
       void BFMT::improvedExpandTreeFromNode(BiDirMotion *&z, BiDirMotion *&connection_point)
       {
           // Define Opennew and set it to NULL
           BiDirMotionPtrs Open_new;

           // Define Znear as all unexplored nodes in the neighborhood around z
           BiDirMotionPtrs zNear;
           const BiDirMotionPtrs &zNeighborhood = neighborhoods_[z];

           for (auto i : zNeighborhood)
           {
               if (!precomputeNN_)
                   saveNeighborhood(nn_, i);  // nearest neighbors

               if (i->getCurrentSet() == BiDirMotion::SET_UNVISITED)
               {
                   zNear.push_back(i);
               }
           }

           // For each node x in Znear
           for (auto x : zNear)
           {
//               if (!precomputeNN_)
//                   saveNeighborhood(nn_, x);  // nearest neighbors

               // Define Xnear as all frontier nodes in the neighborhood around the unexplored node x
               BiDirMotionPtrs xNear;
               const BiDirMotionPtrs &xNeighborhood = neighborhoods_[x];
               for (auto j : xNeighborhood)
               {
                   if (j->getCurrentSet() == BiDirMotion::SET_OPEN)
                   {
                       xNear.push_back(j);
                   }
               }
               // Find the node in Xnear with minimum cost-to-come in the current tree
               BiDirMotion *xMin = nullptr;
              base::Cost cMin(std::numeric_limits<double>::infinity());
              for (auto &j : xNear)
              {
                  // check if node costs are smaller than minimum
//                   double cNew = j->getCost().value() + distanceFunction(j, x);
                  const base::State *s = j->getState();
                  const base::Cost dist = opt_->motionCost(s, x->getState());
                  const base::Cost cNew = opt_->combineCosts(j->getCost(), dist);

//                   if (cNew < cMin)
//                   {
//                       xMin = j;
//                       cMin = cNew;
//                   }
                  if (opt_->isCostBetterThan(cNew, cMin))
                  {
                      xMin = j;
                      cMin = cNew;
                  }
              }

              //4.5 2021 modify new state rejection based on inadmissiable Heuristic
              base::Cost improvedsolutionHeuristic;

//               if(tree_==FWD)
//               {
              if(connection_point!=nullptr)
              {
              const base::Cost costToGo =
                    opt_->motionCostHeuristic(x->getState(), heurGoalState_[tree_]);

              improvedsolutionHeuristic = opt_->combineCosts(cMin,costToGo);
//               std::cout<<"cost to go"<<costToGo<<std::endl;
              }
//               }
//               else
//               {
//                   const base::Cost costToGo =
//                            x->hcost_[REV];
//                   if(connection_point!=nullptr)
//                      improvedsolutionHeuristic = opt_->combineCosts(cMin,costToGo);
//               }

              // xMin was found
              if (xMin != nullptr&& opt_->isCostBetterThan(improvedsolutionHeuristic,lastCost_)||
                    (xMin != nullptr&& connection_point == nullptr))
               {
                   bool collision_free = false;
                   if (cacheCC_)
                   {
                       if (!xMin->alreadyCC(x))
                       {
                           collision_free = si_->checkMotion(xMin->getState(), x->getState());
                           ++collisionChecks_;
                           // Due to FMT3* design, it is only necessary to save unsuccesful
                           // connection attemps because of collision
                           if (!collision_free)
                               xMin->addCC(x);
                       }
                   }
                   else
                   {
                       ++collisionChecks_;
                       collision_free = si_->checkMotion(xMin->getState(), x->getState());
                   }

                   if (collision_free)
                   {  // motion between yMin and x is obstacle free
                       // add edge from xMin to x
                       x->setParent(xMin);
                       x->setCost(base::Cost(cMin));
                       xMin->getChildren().push_back(x);

                       if (heuristics_)
                           x->setHeuristicCost(opt_->motionCostHeuristic(x->getState(), heurGoalState_[tree_]));

                       // check if new node x is in the other tree; if so, save result
                       if (x->getOtherSet() != BiDirMotion::SET_UNVISITED)
                       {
                           if (connection_point == nullptr)
                           {
                               connection_point = x;

                               connect_motion_=connection_point;//modify
//                               std::cout << "connect_motion has been first found:  " <<
//                                        connection_point->cost_[FWD].value()+connection_point->cost_[REV].value()
//                                         <<" fwd_cost: "<<connection_point->cost_[FWD].value()
//                                        <<" rev_cost: "<<connection_point->cost_[REV].value()<< std::endl;//test code

                               if (termination_ == FEASIBILITY)
                               {
                                   break;
                               }
                           }
                           else
                           {
                               if ((connection_point->cost_[FWD].value() + connection_point->cost_[REV].value()) >
                                   (x->cost_[FWD].value() + x->cost_[REV].value()))
                               {
                                   connection_point = x;

                                   connect_motion_=connection_point;//modify
//                                   connectchange_ = true;
//                                   std::cout << "connect_motion has been updated: " <<
//                                             connection_point->cost_[FWD].value()+connection_point->cost_[REV].value()
//                                             <<" fwd_cost: "<<connection_point->cost_[FWD].value()
//                                            <<" rev_cost: "<<connection_point->cost_[REV].value()<< std::endl;//test code
                               }
                           }
                       }

                       Open_new.push_back(x);                      // add x to Open_new
                       x->setCurrentSet(BiDirMotion::SET_CLOSED);  // remove x from Unvisited
                   }
               }
               const unsigned int xNeighborhoodSize = xNeighborhood.size();

               //modify begin
               if((connection_point!=nullptr) && (x->getCurrentSet() == BiDirMotion::SET_CLOSED))
      //        if (x->getSetType() == Motion::SET_CLOSED)
              {
                  std::vector<BiDirMotion *> hNear; // save all vertices near x that in open set
                  hNear.reserve(xNeighborhoodSize); // 
                  std::vector<base::Cost> costs; //save all rewire costs through x for vertices in hNear
                  std::vector<base::Cost> incCosts; // save straight line cost of x to h in hNear
                  std::vector<std::size_t> sortedCostIndices; // sorted indices
                  CostIndexCompare compareFn(costs, *opt_); // 

                  for (unsigned int i = 0; i < xNeighborhoodSize; ++i) // 
                  {
                      if ( (xNeighborhood[i]->getCurrentSet() == BiDirMotion::SET_OPEN)
                           && ( xNeighborhood[i]->getParent() != x->getParent() ) ) // exclued vertex that share same parents with x
                      {
                          hNear.push_back(xNeighborhood[i]);
                      }
                  }
                  if (costs.size() < hNear.size()) // expand vector size
                  {
                      costs.resize(hNear.size());
                      incCosts.resize(hNear.size());
                      sortedCostIndices.resize(hNear.size());
                  }
                  for (unsigned int i = 0; i < hNear.size(); ++i) // calculate all costs of hNear go through x
                  {
                      incCosts[i] = opt_->motionCost(x->getState(), hNear[i]->getState());
                      costs[i] = opt_->combineCosts(x->getCost(), incCosts[i]);
                  }
                  for (std::size_t i = 0; i < hNear.size(); ++i) // 将hNear容器中向量和costs容器中向量设定关联索引
                  {
                      sortedCostIndices[i] = i;
                  }
                  std::sort(sortedCostIndices.begin(), sortedCostIndices.begin() + hNear.size(), compareFn); // hNear容器中向量和costs容器中向量顺序不变，sortedCostIndices容器中的关联索引i改变，i以重布线后的h集合cost值排序，即原sortedCostIndices[1] = 1、sortedCostIndices[2] = 2、sortedCostIndices[3] = 3,排序后sortedCostIndices[1] = 2、sortedCostIndices[2] = 3、sortedCostIndices[3] = 1,231分别值hNear容器中向量和costs容器中向量索引
                  for (std::vector<std::size_t>::const_iterator i = sortedCostIndices.begin();
                       i != sortedCostIndices.begin() + hNear.size(); ++i) // check every vertex in hNear with cost order
                  {
                      if (opt_->isCostBetterThan(costs[*i], hNear[*i]->getCost())) // if rewire could improve costs, execute collision checks 
                      {
                          BiDirMotion *hhNear = hNear[*i]; // vertex that considered for rewiring
                          bool collision_free = false;
                          if (cacheCC_) // cacheCC 
                          {
                              if (!x->alreadyCC(hhNear)) // 如果x与其最优父节点没进行过碰撞检测
                              {
                                  collision_free = si_->checkMotion(hhNear->getState(), x->getState()); // CC
                                  ++collisionChecks_; // count collision checks
                                  // Due to FMT* design, it is only necessary to save unsuccesful
                                  // connection attemps because of collision
                                  if (!collision_free) // 
                                      x->addCC(hhNear); // cache CC results
                              }
                          }
                          else // normal version
                          {
                              ++collisionChecks_; 
                              collision_free = si_->checkMotion(hhNear->getState(), x->getState()); 
                          }

                          if (collision_free) // if collision-free, execute rewire
                          {  // remove hh from hh's parent's children list
                              std::vector<BiDirMotion*>::iterator iter = std::find(hhNear->getParent()->getChildren().begin()
                                                                              ,hhNear->getParent()->getChildren().end()
                                                                              ,hhNear);
                              //vector< int >::iterator iter=std::find(v.begin(),v.end(),num_to_find); 
                              if(iter != hhNear->getParent()->getChildren().end())//if hhNear is found，delete hhNear from hhNear->getParent()->getChildren()
                              {
                                  hhNear->getParent()->getChildren().erase(iter);
                              }
                              hhNear->setParent(x); // set x as hhNear's new parent
                              hhNear->setCost(costs[*i]); // set new connection costs for hhNear
                              x->getChildren().push_back(hhNear); // put hhNear to x's children list
                              updateChildCosts(hhNear); // update cost for every child nodes of hhNear
                          }
                      }
                  }
              }//modify end

           }  // End "for x in Znear"

           // Remove motion z from binary heap and map
           BiDirMotionBinHeap::Element *zElement = Open_elements[tree_][z];
           Open_[tree_].remove(zElement);
           Open_elements[tree_].erase(z);
           z->setCurrentSet(BiDirMotion::SET_CLOSED);

           // add nodes in Open_new to Open
           for (auto &i : Open_new)
           {
               Open_elements[tree_][i] = Open_[tree_].insert(i);
               i->setCurrentSet(BiDirMotion::SET_OPEN);
           }
       }
// plan function
       bool BFMT::plan(BiDirMotion *x_init, BiDirMotion *x_goal, BiDirMotion *&connection_point,
                       const base::PlannerTerminationCondition &ptc)
       {
           // Expand the trees until reaching the termination condition
           bool earlyFailure = false;
           bool success = false;
           int i = 1;
           int in = 1;
//           base::Cost costThreshold(1660); // set path length optimization threshold
//           opt_->setCostThreshold(costThreshold);
           bool sufficientlyShort = false;
           bool firstSuccessful_ = false;
//           bool finalSuccessful_ = false;
          int improvN = numSamples_;
          while(!ptc)
          {

            if(!firstSuccessful_)
            {
               // If pre-computation, find neighborhoods for all N sample nodes plus initial
               // and goal state(s).  Otherwise compute the neighborhoods of the initial and
               // goal states separately and compute the others as needed.
               BiDirMotionPtrs sampleNodes;
               nn_->list(sampleNodes);
               /// \todo This precomputation is useful only if the same planner is used many times.
               /// otherwise is probably a waste of time. Do a real precomputation before calling solve().
               if (precomputeNN_)
               {
                   for (auto &sampleNode : sampleNodes)
                   {
                       saveNeighborhood(nn_, sampleNode);  // nearest neighbors
                   }
               }
               else
               {
                   saveNeighborhood(nn_, x_init);  // nearest neighbors
                   saveNeighborhood(nn_, x_goal);  // nearest neighbors
               }

               // Copy nodes in the sample set to Unvisitedfwd.  Overwrite the label of the initial
               // node with set Open for the forward tree, since it starts in set Openfwd.
               useFwdTree();
                for (auto &sampleNode : sampleNodes)
                {
                    sampleNode->setCurrentSet(BiDirMotion::SET_UNVISITED);
                }
               x_init->setCurrentSet(BiDirMotion::SET_OPEN);

               // Copy nodes in the sample set to Unvisitedrev.  Overwrite the label of the goal
               // node with set Open for the reverse tree, since it starts in set Openrev.
               useRevTree();
                for (auto &sampleNode : sampleNodes)
                {
                    sampleNode->setCurrentSet(BiDirMotion::SET_UNVISITED);
                }
               x_goal->setCurrentSet(BiDirMotion::SET_OPEN);

//                // Expand the trees until reaching the termination condition
//                bool earlyFailure = false;
//                bool success = false;

               useFwdTree();
               BiDirMotion *z = x_init;

//               std::cout << "the " << i << " time planning begin " << std::endl;
               while (!success)
               {
                   ++iterations_;
                   improvedExpandTreeFromNode(z, connection_point);
//                   drawPathe();//单点绘制modify

                   // Check if the algorithm should terminate.  Possibly redefines connection_point.
                   if (termination(z, connection_point, ptc))
                   {
                       success = true;
//                       successful_= true;
                       firstSuccessful_ = true;
                       traceSolutionPathThroughTree(connection_point);
                       OMPL_DEBUG("first path cost: %f", lastCost_.value());
                       iterationcost_ = lastCost_;
                       ++iterTimes_;
//                       drawPathe();//
                   }

                   else// dynamic insert new samples to find a inital path
                   {
                       if (Open_[tree_].empty())  // If this heap is empty...
                       {
                            insertNewSampleInOpen(ptc);
//                           drawPathe();
                       }

                       // This function will be always reached with at least one state in one heap.
                       // However, if ptc terminates, we should skip this.
                       if (!ptc)
                           chooseTreeAndExpansionNode(z);
                       else
                           return true;
                   }
               }
               i = i+1;
            }

            else
            {
                if (iterTimes_== terminatetime_)
                {
                    break;
                }
//                std::cout<< "the "<<in <<" times improving planning begin " << std:: endl;
                Open_[FWD].clear();
                Open_[REV].clear();
                Open_elements[FWD].clear();
                Open_elements[REV].clear();
                neighborhoods_.clear();
                BiDirMotion *z; // expansion node
                bool improvesucess = false;
                useFwdTree();
//                                    useRevTree();
//                                    if(tree_==FWD)
//                                        std::cout<<"current tree type:  FWD"<< std:: endl;
//                                    else
//                                        std::cout<<"current tree type:  REV"<< std:: endl;

                int ii = 1;
                std::vector<BiDirMotion *> allMotions;
                nn_->list(allMotions); // save all vertices to allMotions
                nn_->clear();
                for (BiDirMotion *everyMotion : allMotions) 
                {
                    base::Cost solutionHeuristic;
                    if (ii != 1)
                    { // calculate heuristic cost of every motion solutionHeuristic
                        base::Cost costToCome;

                        // Start with infinite cost
                        costToCome = opt_->infiniteCost();

                        // Find the min from each start
                        costToCome = opt_->betterCost(costToCome,
                                    opt_->motionCost(x_init->getState(),everyMotion->getState()));  // lower-bounding cost from the start to the state

                        const base::Cost costToGo =
                                opt_->costToGo(everyMotion->getState(), pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
                        solutionHeuristic = opt_->combineCosts(costToCome, costToGo);            // add the two costs
                    }

// set start vertex for searching
                    if (ii == 1)
                    { 
                        Open_elements[FWD][everyMotion] = Open_[FWD].insert(everyMotion);
                        everyMotion->currentSet_[FWD] = BiDirMotion::SET_OPEN;
//                             z = everyMotion;  // z <-- xinit
                        nn_->add(everyMotion); // add start vertex to nn_ structure
                    }
                    else
                    {
                        if(opt_->isCostBetterThan(solutionHeuristic,lastCost_))
                             nn_->add(everyMotion);

                        if((everyMotion->getParent() != nullptr)
                                          && (opt_->isCostBetterThan(solutionHeuristic,lastCost_)))
                        {
                            Open_elements[FWD][everyMotion] = Open_[FWD].insert(everyMotion);
                            everyMotion->currentSet_[FWD] = BiDirMotion::SET_OPEN;
                        }
                        useRevTree();
                        if((everyMotion->getParent() != nullptr)
                                          && (opt_->isCostBetterThan(solutionHeuristic,lastCost_)))
                        {
//                                std::cout<<"add into reverse tree"<< std:: endl;
                            Open_elements[REV][everyMotion] = Open_[REV].insert(everyMotion);
                            everyMotion->currentSet_[REV] = BiDirMotion::SET_OPEN;
                        }
                        swapTrees();
                    }
                    ii = ii + 1;
                }
                 ++iterTimes_;

                setBatchNumSamples(improvN); // set sample counts of every batch
//                std::cout<<"batch samples: "<< batchnumSamples_ << "nnsize: "<< nn_->size()<<std::endl;
                if (sampler_)
                    sampler_.reset();
                sampleFree(nn_,ptc);
                improvN = improvN*(increaseFactor+1);
                OMPL_INFORM("Improving planning with %u states already in datastructure", nn_->size());
                // Calculate the nearest neighbor search radius
                if (nearestK_)
                {
                    NNk_ = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                                     (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                                     log((double)nn_->size()));
                    OMPL_DEBUG("Using nearest-neighbors k of %d", NNk_);
                }
                else
                {
                    NNr_ = calculateRadius(si_->getStateDimension(), nn_->size());
                    OMPL_DEBUG("Using radius of %f", NNr_);
                }

                BiDirMotionPtrs sampleNodes;
                nn_->list(sampleNodes);
                /// \todo This precomputation is useful only if the same planner is used many times.
                /// otherwise is probably a waste of time. Do a real precomputation before calling solve().
                if (precomputeNN_)
                {
                    for (auto &sampleNode : sampleNodes)
                    {
                        saveNeighborhood(nn_, sampleNode);  // nearest neighbors
                    }
                }
                else
                {
                    saveNeighborhood(nn_, x_init);  // nearest neighbors
                    saveNeighborhood(nn_, x_goal);  // nearest neighbors
                }
                useFwdTree();
                x_init->setCurrentSet(BiDirMotion::SET_OPEN);
//                x_init->currentSet_[REV] = BiDirMotion::SET_UNVISITED;
//                 nn_->add(x_init);  // S <-- {x_init}

                    // Take the first valid initial state as the forward tree root
//                     Open_elements[FWD][x_init] = Open_[FWD].insert(x_init);
//                    x_init->currentSet_[FWD] = BiDirMotion::SET_OPEN;
                    x_init->cost_[FWD] = opt_->initialCost(x_init->getState());
                    heurGoalState_[1] = x_init->getState();

                useRevTree();
                x_goal->setCurrentSet(BiDirMotion::SET_OPEN);
//                x_goal->currentSet_[FWD] = BiDirMotion::SET_UNVISITED;
//                 nn_->add(x_goal);  // S <-- {x_goal}

                    // Take the first valid goal state as the reverse tree root
                    Open_elements[REV][x_goal] = Open_[REV].insert(x_goal);//x_goal have to be added, since it does not have a parent
//                    x_goal->currentSet_[REV] = BiDirMotion::SET_OPEN;
                    x_goal->cost_[REV] = opt_->terminalCost(x_goal->getState());
                    heurGoalState_[0] = x_goal->getState();

                useFwdTree();
                z = x_init;
                double lastMotionCostValue = lastCost_.value();
                while (!improvesucess)
                {
                    ++iterations_;
                    improvedExpandTreeFromNode(z, connection_point);
                    traceSolutionPathThroughTree(connection_point);
//                    drawPathe();//单点绘制modify
                    sufficientlyShort = opt_->isSatisfied(lastCost_);
                    if ( (lastMotionCostValue - lastCost_.value()) > 0)/*update cost if better solution is found*/
                    {  // draw pic 画图，并实时追踪改变的求解路径
                        lastMotionCostValue = lastCost_.value();
//                            traceSolutionPathThroughTree(connection_point);
//                            drawPathe();
//                        lastCost=opt_->combineCosts(connection_point->cost_[FWD],connection_point->cost_[REV]);
                        OMPL_DEBUG("%u th search find improving path cost: %f", iterTimes_,lastCost_.value());

                        if (sufficientlyShort)
                        {
                            return false;
                        }
                    }
                    // Check if the algorithm should terminate.  Possibly redefines connection_point.
                    if (termination(z, connection_point, ptc))
                    {
//                        successful_= true;
                        firstSuccessful_ = true;
                        improvesucess = true;
                        traceSolutionPathThroughTree(connection_point);
                        iterationcost_ = lastCost_;
                        OMPL_DEBUG("%u th search find improving path cost: %f", iterTimes_,lastCost_.value());
//                        drawPathe();//绘制最终路径modify
                        in = in + 1;
                    }

                    else
                    {
                    // This function will be always reached with at least one state in one heap.
                    // However, if ptc terminates, we should skip this.
                    if (!ptc)
                        chooseTreeAndExpansionNode(z);
                    else
                        return true;
                    }
                }//while(!improvesuccess)
            }//else
          }//while(!finalsuccessful)
               earlyFailure = false;
               return earlyFailure;
       }
 // iteration
       void ompl::geometric::BFMT::updateChildCosts(BiDirMotion *m)
       {
           for (unsigned int i = 0; i < m->getChildren().size(); ++i)
           {
               base::Cost incCost;
               incCost = opt_->motionCost(m->getState(), m->getChildren()[i]->getState());
               m->getChildren()[i]->setCost(opt_->combineCosts(m->getCost(), incCost));
               updateChildCosts(m->getChildren()[i]);
           }
       }
// Animate the algorithm process with openCV library
//       void ompl::geometric::BFMT::drawPathe()
//       {
//           if(tree_==FWD)
//           {
//           // 显示并输出最终规划路径图像，并输出图像文件
//           Mat envImageCopy; // 复制后的图像变量
//           Mat envImageResize; // 修改图像尺寸后点图像变量
//           envImage_.copyTo(envImageCopy); // 复制图像
////            useFwdTree();
//           BiDirMotionPtrs  drawMotions;
//           nn_->list(drawMotions); // 将树上的数据存入testMotions中
//           for (BiDirMotion *drawMotion : drawMotions) // 遍历树上点数据
//           {
//               double stateX = drawMotion->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0]; // 树上数据点的x，y值
//               double stateY = drawMotion->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//               circle( envImageCopy,Point( stateX, stateY ),6,Scalar(0, 0, 255 ),-1,6 ); //画树上状态点
//               if (drawMotion->getParent() != nullptr) // 如果父节点不空，即不是起点
//               { // 打印父节点，画点和线
//                   double parentSateX = drawMotion->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                   double parentSateY = drawMotion->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                   if(tree_==FWD)
//                   {
//                       circle( envImageCopy,Point( parentSateX, parentSateY ),6,Scalar( 0, 0, 255 ),-1,6 );
//                       line( envImageCopy, Point( parentSateX, parentSateY ), Point( stateX, stateY ), Scalar( 0, 255, 0 ), 2, CV_AA );
//                   }
//                   else
//                   {
//                       circle( envImageCopy,Point( parentSateX, parentSateY ),6,Scalar(0, 0, 255),-1,6 );
//                       line( envImageCopy, Point( parentSateX, parentSateY ), Point( stateX, stateY ), Scalar( 125, 0, 125 ), 2, CV_AA );
//                   }
//                   }
//           }
//           swapTrees();
//           for (BiDirMotion *drawMotion : drawMotions) // 遍历树上点数据
//           {
//               double stateX = drawMotion->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0]; // 树上数据点的x，y值
//               double stateY = drawMotion->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//               circle( envImageCopy,Point( stateX, stateY ),6,Scalar(0, 0, 255 ),-1,8 ); //画树上状态点
//               if (drawMotion->getParent() != nullptr) // 如果父节点不空，即不是起点
//               { // 打印父节点，画点和线
//                   double parentSateX = drawMotion->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                   double parentSateY = drawMotion->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                   if(tree_==FWD)
//                   {
//                       circle( envImageCopy,Point( parentSateX, parentSateY ),6,Scalar( 0, 0, 255 ),-1,6 );
//                       line( envImageCopy, Point( parentSateX, parentSateY ), Point( stateX, stateY ), Scalar( 0, 255, 0 ), 2, CV_AA );
//                   }
//                   else
//                   {
//                       circle( envImageCopy,Point( parentSateX, parentSateY ),6,Scalar(0, 0, 255),-1,6 );
//                       line( envImageCopy, Point( parentSateX, parentSateY ), Point( stateX, stateY ), Scalar( 125, 0, 125 ), 2, CV_AA );
//                   }
//                   }
//           }
//           swapTrees();
//           // 画起点和终点, circle, 参数：图像、位置、半径、BRG颜色、填满圆、8联通绘图方式
//           double startStateX = drawMotions[0]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//           double startStateY = drawMotions[0]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//           double goalStateX = goalState_->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//           double goalStateY = goalState_->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];

//           circle( envImageCopy,Point( startStateX, startStateY ),15,Scalar( 128, 128, 0 ),-1,8 );
//           circle( envImageCopy,Point( goalStateX, goalStateY ),15,Scalar( 0, 255, 165 ),-1,8 );

//           // 输出中间图像
//           cv::namedWindow("path", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO); // 配置OPENCV窗口
//           cv::resize(envImageCopy, envImageResize, cv::Size(), 0.65, 0.65); // 改变窗口大小
//           cv::imshow("path", envImageResize); // 显示窗口图像
//           if (connect_motion_ != nullptr)
//           {
//               double connectStateX = connect_motion_->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//               double connectStateY = connect_motion_->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//               circle( envImageCopy,Point( connectStateX, connectStateY ),15,Scalar( 255, 0, 165 ),-1,8 );
//               if(connectchange_)
//               {
//                  cv::waitKey(500);
//                  connectchange_ = false;
//               }
//               for (int i = 0; i <= pwdPathSize_ - 1; ++i)
//               { // 获取路径点x，y值，画最终规划路径线
//                   double pathSateX = pwdpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                   double pathSateY = pwdpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                   if (pwdpath_[i]->getParent() != nullptr)
//                   {
//                       double pathParentSateX = pwdpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                       double pathParentSateY = pwdpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                       line( envImageCopy, Point( pathParentSateX, pathParentSateY ), Point( pathSateX, pathSateY ), Scalar( 255, 0, 0 ), 3, CV_AA );
//                   }
//               }
//               swapTrees();
//               for (int i = 0; i <= revPathSize_ - 1; ++i)
//               { // 获取路径点x，y值，画最终规划路径线
//                   double pathSateX = revpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                   double pathSateY = revpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                   if (revpath_[i]->getParent() != nullptr)
//                   {
//                       double pathParentSateX = revpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                       double pathParentSateY = revpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                       line( envImageCopy, Point( pathParentSateX, pathParentSateY ), Point( pathSateX, pathSateY ), Scalar( 255, 0, 0 ), 3, CV_AA );
//                   }
//               }
//               swapTrees();

//               std::string graph = "/home/wangkuan/workspace/Documents/Graph/finalPath.ppm";
//               cv::namedWindow("path", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
//               cv::resize(envImageCopy, envImageResize, cv::Size(), 0.65, 0.65);
//               cv::imshow("path", envImageResize);

//               if(finalSuccessful_)//这里导致无法绘制ｒｅｖ树
//               {
//                   for (int i = 0; i <= mPathSize_ - 1; ++i)
//                   { // 获取路径点x，y值，画最终规划路径线
//                       double pathSateX = mpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                       double pathSateY = mpath_[i]->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                       if (mpath_[i]->getParent() != nullptr)
//                       {
//                           double pathParentSateX = mpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
//                           double pathParentSateY = mpath_[i]->getParent()->getState()->as<ompl::base::RealVectorStateSpace::StateType>()->values[1];
//                           line( envImageCopy, Point( pathParentSateX, pathParentSateY ), Point( pathSateX, pathSateY ), Scalar( 255, 0, 0 ), 3, CV_AA );
//                       }
//                   }
//                   std::string graph = "/home/wangkuan/workspace/Documents/Graph/finalPath.ppm";
//                   cv::namedWindow("path", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
//                   cv::resize(envImageCopy, envImageResize, cv::Size(), 0.65, 0.65);
//                   cv::imshow("path", envImageResize);
//                   cv::waitKey(5000);
//               }
//           }
//           cv::waitKey(1); // 停留1毫秒
//       }
//       }

// insert one sample to the open set
       void BFMT::insertNewSampleInOpen(const base::PlannerTerminationCondition &ptc)
       {
           // Sample and connect samples to tree only if there is
           // a possibility to connect to unvisited nodes.
           std::vector<BiDirMotion *> nbh;
           std::vector<base::Cost> costs;
           std::vector<base::Cost> incCosts;
           std::vector<std::size_t> sortedCostIndices;

           // our functor for sorting nearest neighbors
           CostIndexCompare compareFn(costs, *opt_);

           auto *m = new BiDirMotion(si_, &tree_);
           while (!ptc && Open_[tree_].empty())  //&& oneSample)
           {
               // Get new sample and check whether it is valid.
               sampler_->sampleUniform(m->getState());
               ++sampleCount_;

//               ++extendCount_;
               if (!si_->isValid(m->getState()))
                   continue;
               if(si_->isValid(m->getState()))
               {
                   ++iterations_;
                   ++validsample_;
                   ++extendCount_;
               }

               // Get neighbours of the new sample.
               std::vector<BiDirMotion *> yNear;
               if (nearestK_)
                   nn_->nearestK(m, NNk_, nbh);
               else
                   nn_->nearestR(m, NNr_, nbh);

               yNear.reserve(nbh.size());
               for (auto &j : nbh)
               {
                   if (j->getCurrentSet() == BiDirMotion::SET_CLOSED)
                   {
                       if (nearestK_)
                       {
                           // Only include neighbors that are mutually k-nearest
                           // Relies on NN datastructure returning k-nearest in sorted order
                           const base::Cost connCost = opt_->motionCost(j->getState(), m->getState());
                           const base::Cost worstCost =
                               opt_->motionCost(neighborhoods_[j].back()->getState(), j->getState());

                           if (opt_->isCostBetterThan(worstCost, connCost))
                               continue;
                           yNear.push_back(j);
                       }
                       else
                           yNear.push_back(j);
                   }
               }

               // Sample again if the new sample does not connect to the tree.
               if (yNear.empty())
                   continue;

               // cache for distance computations
               //
               // Our cost caches only increase in size, so they're only
               // resized if they can't fit the current neighborhood
               if (costs.size() < yNear.size())
               {
                   costs.resize(yNear.size());
                   incCosts.resize(yNear.size());
                   sortedCostIndices.resize(yNear.size());
               }

               // Finding the nearest neighbor to connect to
               // By default, neighborhood states are sorted by cost, and collision checking
               // is performed in increasing order of cost
               //
               // calculate all costs and distances
               for (std::size_t i = 0; i < yNear.size(); ++i)
               {
                   incCosts[i] = opt_->motionCost(yNear[i]->getState(), m->getState());
                   costs[i] = opt_->combineCosts(yNear[i]->getCost(), incCosts[i]);
               }

               // sort the nodes
               //
               // we're using index-value pairs so that we can get at
               // original, unsorted indices
               for (std::size_t i = 0; i < yNear.size(); ++i)
                   sortedCostIndices[i] = i;
               std::sort(sortedCostIndices.begin(), sortedCostIndices.begin() + yNear.size(), compareFn);

               // collision check until a valid motion is found
               for (std::vector<std::size_t>::const_iterator i = sortedCostIndices.begin();
                    i != sortedCostIndices.begin() + yNear.size(); ++i)
               {
                   ++collisionChecks_;

                   if (si_->checkMotion(yNear[*i]->getState(), m->getState()))
                   {
                       const base::Cost incCost = opt_->motionCost(yNear[*i]->getState(), m->getState());
                       m->setParent(yNear[*i]);
                       yNear[*i]->getChildren().push_back(m);
                       m->setCost(opt_->combineCosts(yNear[*i]->getCost(), incCost));
                       m->setHeuristicCost(opt_->motionCostHeuristic(m->getState(), heurGoalState_[tree_]));
                       m->setCurrentSet(BiDirMotion::SET_OPEN);
                       Open_elements[tree_][m] = Open_[tree_].insert(m);

                       nn_->add(m);
                       saveNeighborhood(nn_, m);
                       updateNeighborhood(m, nbh);

                       break;
                   }
               }
           }  // While Open_[tree_] empty
       }

       bool BFMT::termination(BiDirMotion *&z, BiDirMotion *&connection_point,
                              const base::PlannerTerminationCondition &ptc)
       {
           bool terminate = false;

           switch (termination_)
           {
               case FEASIBILITY:
                   // Test if a connection point was found during tree expansion
                   return (connection_point != nullptr || ptc);
                   break;

               case OPTIMALITY:
                   // Test if z is in SET_CLOSED (interior) of other tree
                   if (ptc)
                       terminate = true;
                   else if (z->getOtherSet() == BiDirMotion::SET_CLOSED)
                       terminate = true;

                   break;
           };
           return terminate;
       }

       // Choose exploration tree and node z to expand
       void BFMT::chooseTreeAndExpansionNode(BiDirMotion *&z)
       {
           switch (exploration_)
           {
               case SWAP_EVERY_TIME:
                   if (Open_[(tree_ + 1) % 2].empty())
                       z = Open_[tree_].top()->data;  // Continue expanding the current tree (not empty by exit
                                                      // condition in plan())
                   else
                   {
                       z = Open_[(tree_ + 1) % 2].top()->data;  // Take top of opposite tree heap as new z
                       swapTrees();                             // Swap to the opposite tree
                   }
                   break;

               case CHOOSE_SMALLEST_Z:
                   BiDirMotion *z1, *z2;
                   if (Open_[(tree_ + 1) % 2].empty())
                       z = Open_[tree_].top()->data;  // Continue expanding the current tree (not empty by exit
                                                      // condition in plan())
                   else if (Open_[tree_].empty())
                   {
                       z = Open_[(tree_ + 1) % 2].top()->data;  // Take top of opposite tree heap as new z
                       swapTrees();                             // Swap to the opposite tree
                   }
                   else
                   {
                       z1 = Open_[tree_].top()->data;
                       z2 = Open_[(tree_ + 1) % 2].top()->data;

                       if (z1->getCost().value() < z2->getOtherCost().value())
                           z = z1;
                       else
                       {
                           z = z2;
                           swapTrees();
                       }
                   }
                   break;
           };
       }

       // Trace a path of nodes along a tree towards the root (forward or reverse)
       void BFMT::tracePath(BiDirMotion *z, BiDirMotionPtrs &path)
       {
           BiDirMotion *solution = z;

           while (solution != nullptr)
           {
               path.push_back(solution);
               solution = solution->getParent();
           }
       }

       void BFMT::swapTrees()
       {
           tree_ = (TreeType)((((int)tree_) + 1) % 2);
       }

       void BFMT::updateNeighborhood(BiDirMotion *m, const std::vector<BiDirMotion *> nbh)
       {
           // Neighborhoods are only updated if the new motion is within bounds (k nearest or within r).
           for (auto i : nbh)
           {
               // If CLOSED, that neighborhood won't be used again.
               // Else, if neighhboorhod already exists, we have to insert the node in
               // the corresponding place of the neighborhood of the neighbor of m.
               if (i->getCurrentSet() == BiDirMotion::SET_CLOSED)
                   continue;

               auto it = neighborhoods_.find(i);
               if (it != neighborhoods_.end())
               {
                   if (it->second.empty())
                       continue;

                   const base::Cost connCost = opt_->motionCost(i->getState(), m->getState());
                   const base::Cost worstCost = opt_->motionCost(it->second.back()->getState(), i->getState());

                   if (opt_->isCostBetterThan(worstCost, connCost))
                       continue;

                   // insert the neighbor in the vector in the correct order
                   std::vector<BiDirMotion *> &nbhToUpdate = it->second;
                   for (std::size_t j = 0; j < nbhToUpdate.size(); ++j)
                   {
                       // If connection to the new state is better than the current neighbor tested, insert.
                       const base::Cost cost = opt_->motionCost(i->getState(), nbhToUpdate[j]->getState());
                       if (opt_->isCostBetterThan(connCost, cost))
                       {
                           nbhToUpdate.insert(nbhToUpdate.begin() + j, m);
                           break;
                       }
                   }
               }
           }
       }
   }  // End "geometric" namespace
}  // End "ompl" namespace


