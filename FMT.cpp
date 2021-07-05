/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2013, Autonomous Systems Laboratory, Stanford University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Stanford University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Ashley Clark (Stanford) and Wolfgang Pointner (AIT) */
/* Co-developers: Brice Rebsamen (Stanford), Tim Wheeler (Stanford)
                  Edward Schmerling (Stanford), and Javier V. Gómez (UC3M - Stanford)*/
/* Algorithm design: Lucas Janson (Stanford) and Marco Pavone (Stanford) */
/* Acknowledgements for insightful comments: Oren Salzman (Tel Aviv University),
 *                                           Joseph Starek (Stanford) */

#include <limits>
#include <iostream>


#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <ompl/datastructures/BinaryHeap.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/fmt/FMT.h>

#include <stdio.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/samplers/InformedStateSampler.h>

ompl::geometric::FMT::FMT(const base::SpaceInformationPtr &si)
  : base::Planner(si, "FMT")
{
    // An upper bound on the free space volume is the total space volume; the free fraction is estimated in sampleFree
    freeSpaceVolume_ = si_->getStateSpace()->getMeasure();
    lastGoalMotion_ = nullptr;

    specs_.approximateSolutions = false;
    specs_.directed = false;

    ompl::base::Planner::declareParam<unsigned int>("num_samples", this, &FMT::setNumSamples, &FMT::getNumSamples,
                                                    "10:10:1000000");
    ompl::base::Planner::declareParam<double>("radius_multiplier", this, &FMT::setRadiusMultiplier,
                                              &FMT::getRadiusMultiplier, "0.1:0.05:50.");
    ompl::base::Planner::declareParam<bool>("nearest_k", this, &FMT::setNearestK, &FMT::getNearestK, "0,1");
    ompl::base::Planner::declareParam<bool>("cache_cc", this, &FMT::setCacheCC, &FMT::getCacheCC, "0,1");
    ompl::base::Planner::declareParam<bool>("heuristics", this, &FMT::setHeuristics, &FMT::getHeuristics, "0,1");
    ompl::base::Planner::declareParam<bool>("extended_fmt", this, &FMT::setExtendedFMT, &FMT::getExtendedFMT, "0,1");
    ompl::base::Planner::declareParam<double>("batchfactor", this, &FMT::setBatchFactor, &FMT::getBatchFactor, "0.1:0.05:50.");

    addPlannerProgressProperty("best cost REAL", [this] { return std::to_string(bestCost().value()); });
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
                               });//引号里面不能有特殊字符，如＊号
    addPlannerProgressProperty("iteration times INTEGER", [this]
                               {
                                   return std::to_string(numIterTimes());
                               });
    addPlannerProgressProperty("validsample INTEGER", [this]
                               {
                                   return std::to_string(validsamples());
                               });
}

ompl::geometric::FMT::~FMT()
{
    freeMemory();
}

void ompl::geometric::FMT::setup()
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
        Open_.getComparisonOperator().opt_ = opt_.get();
        Open_.getComparisonOperator().heuristics_ = heuristics_;

        if (!nn_)
            nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
        nn_->setDistanceFunction([this](const Motion *a, const Motion *b)
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

void ompl::geometric::FMT::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        motions.reserve(nn_->size());
        nn_->list(motions);
        for (auto &motion : motions)
        {
            si_->freeState(motion->getState());
            delete motion;
        }
    }
}

void ompl::geometric::FMT::clear()
{
    Planner::clear();
    lastGoalMotion_ = nullptr;
    sampler_.reset();
//    numSamples_ = 1000;//这里必须重新设定，否则会无限增大
//    increaseFactor = 1;
    freeMemory();
    if (nn_)
        nn_->clear();
    Open_.clear();
    neighborhoods_.clear();

    collisionChecks_ = 0;
    iterations_ = 0;
    bestCost_ = base::Cost(std::numeric_limits<double>::quiet_NaN());
    sampleCount_ = 0;
    validsample_ = 0;
    extendCount_ = 0;
    iterTimes_ = 0;
//    numSamples_ = 500; // change depend need
}

void ompl::geometric::FMT::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);
    std::vector<Motion *> motions;
    nn_->list(motions);

    if (lastGoalMotion_ != nullptr)
        data.addGoalVertex(base::PlannerDataVertex(lastGoalMotion_->getState()));

    unsigned int size = motions.size();
    for (unsigned int i = 0; i < size; ++i)
    {
        if (motions[i]->getParent() == nullptr)
            data.addStartVertex(base::PlannerDataVertex(motions[i]->getState()));
        else
            data.addEdge(base::PlannerDataVertex(motions[i]->getParent()->getState()),
                         base::PlannerDataVertex(motions[i]->getState()));
    }
}

void ompl::geometric::FMT::saveNeighborhood(Motion *m)
{
    // Check to see if neighborhood has not been saved yet
    if (neighborhoods_.find(m) == neighborhoods_.end())
    {
        std::vector<Motion *> nbh;
        if (nearestK_)
            nn_->nearestK(m, NNk_, nbh);
        else
            nn_->nearestR(m, NNr_, nbh);
        if (!nbh.empty())
        {
            // Save the neighborhood but skip the first element, since it will be motion m
            neighborhoods_[m] = std::vector<Motion *>(nbh.size() - 1, nullptr);
            std::copy(nbh.begin() + 1, nbh.end(), neighborhoods_[m].begin());
        }
        else
        {
            // Save an empty neighborhood
            neighborhoods_[m] = std::vector<Motion *>(0);
        }
    }  // If neighborhood hadn't been saved yet
}

// Calculate the unit ball volume for a given dimension
double ompl::geometric::FMT::calculateUnitBallVolume(const unsigned int dimension) const
{
    if (dimension == 0)
        return 1.0;
    if (dimension == 1)
        return 2.0;
    return 2.0 * boost::math::constants::pi<double>() / dimension * calculateUnitBallVolume(dimension - 2);
}

double ompl::geometric::FMT::calculateRadius(const unsigned int dimension, const unsigned int n) const
{
    double a = 1.0 / (double)dimension;
    double unitBallVolume = calculateUnitBallVolume(dimension);

    return radiusMultiplier_ * 2.0 * std::pow(a, a) * std::pow(freeSpaceVolume_ / unitBallVolume, a) *
           std::pow(log((double)n) / (double)n, a);
}

void ompl::geometric::FMT::sampleFree(const base::PlannerTerminationCondition &ptc)
{
    if (lastGoalMotion_ == nullptr)
    {
        unsigned int nodeCount = 0;
        unsigned int sampleAttempts = 0;
        auto *motion = new Motion(si_);

        // Sample numSamples_ number of nodes from the free configuration space
        while (nodeCount < numSamples_ && !ptc)
        {
            sampler_->sampleUniform(motion->getState());
            ++sampleCount_;
            sampleAttempts++;

            bool collision_free = si_->isValid(motion->getState());

            if (collision_free)
            {
                nodeCount++;
                nn_->add(motion);
                motion = new Motion(si_);
                ++validsample_;
            }  // If collision free
        }      // While nodeCount < numSamples
        si_->freeState(motion->getState());
        delete motion;

        // 95% confidence limit for an upper bound for the true free space volume
        freeSpaceVolume_ = boost::math::binomial_distribution<>::find_upper_bound_on_p(sampleAttempts, nodeCount, 0.05) *
                si_->getStateSpace()->getMeasure();
    }
    else {
        base::InformedSamplerPtr infSampler_; // infomed采样器，单点采样
        infSampler_.reset();  // informed 采样重置
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
        auto *motion = new Motion(si_);

        // Sample numSamples_ number of nodes from the free configuration space
        while (nodeCount < batchnumSamples_ && !ptc)
        {
            infSampler_->sampleUniform(motion->getState(), lastGoalMotion_->getCost());
            sampleAttempts++;
            ++sampleCount_;

            bool collision_free = si_->isValid(motion->getState());

            if (collision_free)
            {
                nodeCount++;
                nn_->add(motion);
                motion = new Motion(si_);
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

void ompl::geometric::FMT::assureGoalIsSampled(const ompl::base::GoalSampleableRegion *goal)
{
    // Ensure that there is at least one node near each goal
    while (const base::State *goalState = pis_.nextGoal())
    {
        auto *gMotion = new Motion(si_);
        si_->copyState(gMotion->getState(), goalState);

        std::vector<Motion *> nearGoal;
        nn_->nearestR(gMotion, goal->getThreshold(), nearGoal);

        // If there is no node in the goal region, insert one
        if (nearGoal.empty())
        {
            OMPL_DEBUG("No state inside goal region");
            if (si_->getStateValidityChecker()->isValid(gMotion->getState()))
            {
                nn_->add(gMotion);
                goalState_ = gMotion->getState();
            }
            else
            {
                si_->freeState(gMotion->getState());
                delete gMotion;
            }
        }
        else  // There is already a sample in the goal region
        {
            goalState_ = nearGoal[0]->getState();
            si_->freeState(gMotion->getState());
            delete gMotion;
        }
    }  // For each goal
}

ompl::base::PlannerStatus ompl::geometric::FMT::solve(const base::PlannerTerminationCondition &ptc)
{
    if (lastGoalMotion_ != nullptr) // 检验上一次的解是否清空，未清空则返回上次解
    {
        OMPL_INFORM("solve() called before clear(); returning previous solution");
        traceSolutionPathThroughTree(lastGoalMotion_);
        OMPL_DEBUG("Final path cost: %f", lastGoalMotion_->getCost().value());
        return base::PlannerStatus(true, false);
    }
    if (!Open_.empty()) // OPEN集合中如果还有数据，则用CLEAR清空所有数据，为开始求解做准备
    {
        OMPL_INFORM("solve() called before clear(); no previous solution so starting afresh");
        clear();
    }
    checkValidity();
    auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get()); // 获取目标点
    Motion *initMotion = nullptr;  // 起始motion

    if (goal == nullptr) // 检验目标点是否可用
    {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }
    // Add start states to V (nn_) and Open 将起始点放入数据集中，并设定数据类别
    std::vector<Motion *> startMotions_; // 起点缓存
    while (const base::State *st = pis_.nextStart())
    {
        initMotion = new Motion(si_);
        si_->copyState(initMotion->getState(), st);
        Open_.insert(initMotion);
        initMotion->setSetType(Motion::SET_OPEN);
        initMotion->setCost(opt_->initialCost(initMotion->getState()));
        nn_->add(initMotion);  // V <-- {x_init}
        startMotions_.push_back(initMotion);
    }

    if (initMotion == nullptr) // 检验起始MOTION是否可用
    {
        OMPL_ERROR("Start state undefined");
        return base::PlannerStatus::INVALID_START;
    }
    int i = 1;
    int firstN = numSamples_; // 未找到初始解前采样点

    // Sample N free states in the configuration space 设定采样点，确定目标点在采样点中
    if (!sampler_)
        sampler_ = si_->allocStateSampler(); // 开辟采样内存
    setNumSamples(firstN); // 设定采样点个数
    sampleFree(ptc); // 采样
    assureGoalIsSampled(goal); // 保证目标点在采样点中

    OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());
    // Calculate the nearest neighbor search radius
    if (nearestK_)
    {  // 计算K临近点数
        NNk_ = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                         (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                         log((double)nn_->size()));
        OMPL_DEBUG("Using nearest-neighbors k of %d", NNk_);
    }
    else
    {  // 计算R临域搜索半径
        NNr_ = calculateRadius(si_->getStateDimension(), nn_->size());
        OMPL_DEBUG("Using radius of %f", NNr_);
    }
    bool successfulExpansion = false;  // 扩展成功标志
    Motion *z;  // open集合点z，用于不断搜索扩展
    z = initMotion;  // z <-- xinit 第一次计算设定搜索起点
    saveNeighborhood(z);  // 若z之前没在附近点容器中，则存z及对应附近点
//    base::Cost costThreshold(1690); // 设定cost阈值，解的cost值小于1800则中断计算
//    opt_->setCostThreshold(costThreshold);
    bool sufficientlyShort = false;
//    int improvN = increaseFactor*numSamples_;
    int improvN=numSamples_;
    int extendN = numSamples_;

    while (!ptc)
    {

        if (lastGoalMotion_ == nullptr) // 如果没有解，就不断添加采样点，重新采样
        {
            do // 不断单次扩展搜索，直到单次扩展失败或时间到
            {
                ++iterations_;
                if ((goal->isSatisfied(z->getState())))
                { // 若z点到达目标点，相似解距离设定为0,解motion为z，停止单次扩展搜索
                    lastGoalMotion_ = z;
                    bestCost_ = lastGoalMotion_->getCost();
                    break;
                }
                successfulExpansion = expandTreeFromNode(&z); // 由z向四周搜索扩展一次

//modify begin 实现extend的batch版本
                  if(!extendedFMT_ && !successfulExpansion) // 如果规划出路径，改善现存树枝
                   {
                      while (!ptc && Open_.empty())
                      {
//                           std::cout << "wkinger1" << std::endl;
                          neighborhoods_.clear(); // 清除所有点及其附近点map
                          Open_.clear(); // 清除open集合
                          Motion *z; // 设定搜索点

                          int ii = 1;
                          std::vector<Motion *> allMotions;
                          nn_->list(allMotions); // 将所有数据存入allMotions中
                          nn_->clear();
                          for (Motion *everyMotion : allMotions) // 遍历所有点数据
                          {
                              base::Cost solutionHeuristic;
                              if (ii != 1)
                              { // 计算从起点过motion点到目标点的最小cost值
                                  base::Cost costToCome;

                                  // Start with infinite cost
                                  costToCome = opt_->infiniteCost();

                                  // Find the min from each start
                                  for (auto &startMotion : startMotions_)
                                  {
                                      costToCome = opt_->betterCost(
                                                  costToCome, opt_->motionCost(startMotion->getState(),
                                                                               everyMotion->getState()));  // lower-bounding cost from the start to the state
                                  }

                                  const base::Cost costToGo =
                                          opt_->costToGo(everyMotion->getState(), pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
                                  solutionHeuristic = opt_->combineCosts(costToCome, costToGo);            // add the two costs
                              }

                              if (ii == 1)
                              { // 设定搜索起始点
                                  Open_.insert(everyMotion);
                                  everyMotion->setSetType(Motion::SET_OPEN);
                                  z = everyMotion;  // z <-- xinit
                                  nn_->add(everyMotion); // 将数据添加到树中
                              }
                              else if ((ii !=1) && (everyMotion->getParent() != nullptr))
      //                                && (opt_->isCostBetterThan(solutionHeuristic,lastGoalMotion_->getCost())))
                              { // 将上次规划树上有父节点的点，并且该点满足从起点经过该点到终点的cost值小于上次解cost值，将该点放入open集合，并设定open属性，即删除树上在infomed区域外的点
                                  Open_.insert(everyMotion);
                                  everyMotion->setSetType(Motion::SET_OPEN);
                                  nn_->add(everyMotion);
//                                    std::cout << "wkinger2" << std::endl;
                              }
                              else if ((ii !=1) && (everyMotion->getParent() == nullptr))
      //                                && (opt_->isCostBetterThan(solutionHeuristic,lastGoalMotion_->getCost())))
                              { // 将infromed区域中未搜索到的点加入数据中，即删除数据集中在informed之外的点
                                  nn_->add(everyMotion);
//                                    std::cout << "wkinger3" << std::endl;
                              }
                              ii = ii + 1;
                          }
//                           std::cout << "wkinger6" << std::endl;
                          setNumSamples(extendN); // 设定采样点
                          if (sampler_)
                          {
                              sampler_.reset();
                              sampler_ = si_->allocStateSampler();
                          }
//                           std::cout << "wkinger7" << std::endl;
                          sampleFree(ptc);
//                           std::cout << "wkinger8" << std::endl;
                          OMPL_INFORM("%s: extending planning with %u states already in datastructure", getName().c_str(), nn_->size());
                          // Calculate the nearest neighbor search radius
                          if (nearestK_)
                          {  // 计算K临近点数
                              NNk_ = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                                               (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                                               log((double)nn_->size()));
                              OMPL_DEBUG("Using nearest-neighbors k of %d", NNk_);
                          }
                          else
                          {  // 计算R临域搜索半径
                              NNr_ = calculateRadius(si_->getStateDimension(), nn_->size());
                              OMPL_DEBUG("Using radius of %f", NNr_);
                          }

                          bool successfulExpansion = false; // 单次搜索扩展标志
                          saveNeighborhood(z);  // 存z及对应附近点

//                           std::cout << "wkinger4" << std::endl;

                          do// 不断循环扩展搜索
                          {
//                               std::cout << "wkinger5" << std::endl;

//                                              drawPathe();
                              successfulExpansion = extendExpandTreeFromNode(&z);
                              if ((goal->isSatisfied(z->getState())))
                              { // 若z点到达目标点，相似解距离设定为0,解motion为z，停止单次扩展搜索
                                  lastGoalMotion_ = z;
              //                    bestCost_ = lastGoalMotion_->getCost();
                                  break;
                              }
//                                 std::cout << "wkinger6" << std::endl;

                          } while( successfulExpansion && !ptc);// 单次扩展成功并且时间未到继续循环
      //                    OMPL_DEBUG("improving path cost: %f", lastGoalMotion_->getCost().value());
                          OMPL_INFORM("%s: extending Created %u states", getName().c_str(), nn_->size());
                      }

                   }
//modify end extend batch版本
                if (extendedFMT_ && !successfulExpansion)
                {
                    // Apply RRT*-like connections: sample and connect samples to tree
                    std::vector<Motion *> nbh;
                    std::vector<base::Cost> costs;
                    std::vector<base::Cost> incCosts;
                    std::vector<std::size_t> sortedCostIndices;

                    // our functor for sorting nearest neighbors
                    CostIndexCompare compareFn(costs, *opt_);

                    auto *m = new Motion(si_);
                    while (!ptc && Open_.empty())
                    {
                        ++sampleCount_;

//                        ++extendCount_;
                        sampler_->sampleUniform(m->getState());

                        if (!si_->isValid(m->getState()))
                            continue;

                        if(si_->isValid(m->getState()))
                        {
                            ++iterations_;
                            ++extendCount_;
                            ++validsample_;
                        }

                        if (nearestK_)
                            nn_->nearestK(m, NNk_, nbh);
                        else
                            nn_->nearestR(m, NNr_, nbh);

                        // Get neighbours in the tree.
                        std::vector<Motion *> yNear;
                        yNear.reserve(nbh.size());
                        for (auto &j : nbh)
                        {
                            if (j->getSetType() == Motion::SET_CLOSED)
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
                                m->setParent(yNear[*i]);
                                yNear[*i]->getChildren().push_back(m);
                                const base::Cost incCost = opt_->motionCost(yNear[*i]->getState(), m->getState());
                                m->setCost(opt_->combineCosts(yNear[*i]->getCost(), incCost));
                                m->setHeuristicCost(opt_->motionCostHeuristic(m->getState(), goalState_));
                                m->setSetType(Motion::SET_OPEN);

                                nn_->add(m);
                                saveNeighborhood(m);
                                updateNeighborhood(m, nbh);

                                Open_.insert(m);
                                z = m;
                                break;
                            }
                        }
                    }  // while (!ptc && Open_.empty())
                }

            }while( successfulExpansion && !ptc );
            // OMPL_INFORM("%s: Created %u states", getName().c_str(), nn_->size());
            // 有解，规划路径追溯，画图
            if (lastGoalMotion_ != nullptr)
            {
                // Return the path to z, since by definition of planner success, z is in the goal region
//                traceSolutionPathThroughTree(lastGoalMotion_); // 追溯路径
                OMPL_DEBUG("first path cost: %f", lastGoalMotion_->getCost().value());
                            ++iterTimes_;
//                break;
            }  // if solved

        }
        else // 如果规划出路径，改善现存树枝
        {
//            improvsolved_ = true; // 改进求解标志
            neighborhoods_.clear(); // 清除所有点及其附近点map
            Open_.clear(); // 清除open集合
            Motion *z; // 设定搜索点

            int ii = 1;
            std::vector<Motion *> allMotions;
            nn_->list(allMotions); // 将所有数据存入allMotions中
            nn_->clear();
            for (Motion *everyMotion : allMotions) // 遍历所有点数据
            {
                base::Cost solutionHeuristic;
                if (ii != 1)
                { // 计算从起点过motion点到目标点的最小cost值
                    base::Cost costToCome;

                    // Start with infinite cost
                    costToCome = opt_->infiniteCost();

                    // Find the min from each start
                    for (auto &startMotion : startMotions_)
                    {
                        costToCome = opt_->betterCost(
                                    costToCome, opt_->motionCost(startMotion->getState(),
                                                                 everyMotion->getState()));  // lower-bounding cost from the start to the state
                    }

                    const base::Cost costToGo =
                            opt_->costToGo(everyMotion->getState(), pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
                    solutionHeuristic = opt_->combineCosts(costToCome, costToGo);            // add the two costs
                }

                if (ii == 1)
                { // 设定搜索起始点
                    Open_.insert(everyMotion);
                    everyMotion->setSetType(Motion::SET_OPEN);
                    z = everyMotion;  // z <-- xinit
                    nn_->add(everyMotion); // 将数据添加到树中
                }
                else if ((ii !=1) && (everyMotion->getParent() != nullptr)
                        && (opt_->isCostBetterThan(solutionHeuristic,lastGoalMotion_->getCost())))
                { // 将上次规划树上有父节点的点，并且该点满足从起点经过该点到终点的cost值小于上次解cost值，将该点放入open集合，并设定open属性，即删除树上在infomed区域外的点
                    Open_.insert(everyMotion);
                    everyMotion->setSetType(Motion::SET_OPEN);
                    nn_->add(everyMotion);
                }
                else if ((ii !=1) && (everyMotion->getParent() == nullptr)
                        && (opt_->isCostBetterThan(solutionHeuristic,lastGoalMotion_->getCost())))
                { // 将infromed区域中未搜索到的点加入数据中，即删除数据集中在informed之外的点
                    nn_->add(everyMotion);
                }
                ii = ii + 1;
            }
            ++iterTimes_;

            setBatchNumSamples(improvN); // 设定采样点
//            std::cout<<"improvN: "<< improvN << "nnsize: "<< nn_->size()<<std::endl;
            if (sampler_)
                sampler_.reset();
            sampleFree(ptc);
            improvN = improvN*(increaseFactor+1);
            OMPL_INFORM("%s: Improving planning with %u states already in datastructure", getName().c_str(), nn_->size());
            // Calculate the nearest neighbor search radius
            if (nearestK_)
            {  // 计算K临近点数
                NNk_ = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                                 (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                                 log((double)nn_->size()));
                OMPL_DEBUG("Using nearest-neighbors k of %d", NNk_);
            }
            else
            {  // 计算R临域搜索半径
                NNr_ = calculateRadius(si_->getStateDimension(), nn_->size());
                OMPL_DEBUG("Using radius of %f", NNr_);
            }

            bool successfulExpansion = false; // 单次搜索扩展标志
            saveNeighborhood(z);  // 存z及对应附近点
            double lastMotionCostValue = lastGoalMotion_->getCost().value();
            do // 不断循环扩展搜索
            {
                ++iterations_;
                if ((goal->isSatisfied(z->getState())))
                { // 若z点到达目标点，相似解距离设定为0,解motion为z，停止单次扩展搜索
                    bestCost_ = lastGoalMotion_->getCost();
                    break;
                }
                successfulExpansion = expandTreeFromNode(&z);
                if ( (lastMotionCostValue - lastGoalMotion_->getCost().value()) >0 )
                {  // draw pic 画图，并实时追踪改变的求解路径
                    bestCost_ = lastGoalMotion_->getCost();
                    lastMotionCostValue = lastGoalMotion_->getCost().value();
//                    traceSolutionPathThroughTree(lastGoalMotion_);
                    OMPL_DEBUG("improving path cost: %f", lastGoalMotion_->getCost().value());
                    sufficientlyShort = opt_->isSatisfied(lastGoalMotion_->getCost());
                    if (sufficientlyShort)
                    {
                        break;
                    }
                }
            }while( successfulExpansion && !ptc); // 单次扩展成功并且时间未到继续循环
            OMPL_DEBUG("improving path cost: %f", lastGoalMotion_->getCost().value());
            OMPL_INFORM("%s: improving Created %u states", getName().c_str(), nn_->size());
        }//else
        if (sufficientlyShort)
        {
            break;
        }
        i = i+1;
    }//while(!ptc)
    traceSolutionPathThroughTree(lastGoalMotion_);
    return base::PlannerStatus(lastGoalMotion_ != nullptr, false);
}

void ompl::geometric::FMT::traceSolutionPathThroughTree(Motion *goalMotion)
{
    std::vector<Motion *> mpath;
    Motion *solution = goalMotion;

    // Construct the solution path
    while (solution != nullptr)
    {
        mpath.push_back(solution);
        solution = solution->getParent();
    }

    // Set the solution path
    auto path(std::make_shared<PathGeometric>(si_));
    int mPathSize = mpath.size();
    for (int i = mPathSize - 1; i >= 0; --i)
        path->append(mpath[i]->getState());
    pdef_->addSolutionPath(path, false, -1.0, getName());
}

bool ompl::geometric::FMT::extendExpandTreeFromNode(Motion **z)
{
    // Find all nodes that are near z, and also in set Unvisited 寻找z附近所有未访问过的点
    std::vector<Motion *> xNear; // z附近x个状态点，不包含z本身
    const std::vector<Motion *> &zNeighborhood = neighborhoods_[*z]; // z附近的状态点，不包含z本身
    const unsigned int zNeighborhoodSize = zNeighborhood.size(); // z附近状态点个数
    xNear.reserve(zNeighborhoodSize); // 开辟xNear内存空间，大小是z附近状态点个数的的大小

    for (unsigned int i = 0; i < zNeighborhoodSize; ++i) // 遍历z附近所有点，将所有未访问过点放一起
    {
        Motion *x = zNeighborhood[i]; // z附近点中的一个赋值给x
         saveNeighborhood(x); // 获取该点的附近点
        if (x->getSetType() == Motion::SET_UNVISITED) // 如果该附近点未被访问过
        {
//            saveNeighborhood(x); // 获取该点的附近点
            if (nearestK_)
            {
                // Only include neighbors that are mutually k-nearest
                // Relies on NN datastructure returning k-nearest in sorted order
                const base::Cost connCost = opt_->motionCost((*z)->getState(), x->getState());
                const base::Cost worstCost = opt_->motionCost(neighborhoods_[x].back()->getState(), x->getState());

                if (opt_->isCostBetterThan(worstCost, connCost))
                    continue;
                xNear.push_back(x);
            }
            else
                xNear.push_back(x); // 将该点放入xNear，xNear中存的都是z附近未访问过的点
        }
    }

    // For each node near z and in set Unvisited, attempt to connect it to set Open 尝试对z附近未访问过的点放入open集合
    std::vector<Motion *> yNear; // z附近未访问的一个点x的属于open集合的附近点
    std::vector<Motion *> Open_new; // 新加入open集合中的几个点
    const unsigned int xNearSize = xNear.size(); // z附近未访问点数量
    for (unsigned int i = 0; i < xNearSize; ++i) // 遍历z附近未访问点
    {
        Motion *x = xNear[i]; // z附近未访问点中的一个点x

        // Find all nodes that are near x and in set Open 该未访问点x附近所有open集合中的点
        const std::vector<Motion *> &xNeighborhood = neighborhoods_[x]; // 未访问点x附近所有点

        const unsigned int xNeighborhoodSize = xNeighborhood.size(); // x附近点个数
        yNear.reserve(xNeighborhoodSize); // 开辟内存
        for (unsigned int j = 0; j < xNeighborhoodSize; ++j) // 遍历z附近未访问点x附近的所有点
        {
            if (xNeighborhood[j]->getSetType() == Motion::SET_OPEN) // 从中找到属性为open的点
                yNear.push_back(xNeighborhood[j]); // 将x附近的属于open集合的点放到一起
        }

        // Find the lowest cost-to-come connection from Open to x 寻找x附近open集合点到x点的最小cost连接
        base::Cost cMin(std::numeric_limits<double>::infinity()); // cost最小值变量，初始为无穷大
        Motion *yMin = getBestParent(x, yNear, cMin); // 在x附近属性为open的点集合找到x的最佳父状态点，并得到x过最佳父节点的最小cost值
        yNear.clear(); // 清空x附近属性为open的点集合

        // If an optimal connection from Open to x was found 如果x到附近open集合的最佳父状态点存在
        if (yMin != nullptr) // 如果x最优父状态点不空
        {
            bool collision_free = false;
            if (cacheCC_) // 如果碰撞检测缓存标志为真，碰撞检测缓存用于减少碰撞检测，即曾经检测过的两个点不再检测
            {
                if (!yMin->alreadyCC(x)) // 如果x与其最优父节点没进行过碰撞检测
                {
                    collision_free = si_->checkMotion(yMin->getState(), x->getState()); // 进行碰撞检测
                    ++collisionChecks_; // 碰撞次数计数
                    // Due to FMT* design, it is only necessary to save unsuccesful
                    // connection attemps because of collision
                    if (!collision_free) // 如果x与其最优open父节点有碰撞
                        yMin->addCC(x); // 缓存碰撞失败的连接
                }
            }
            else // 碰撞检测缓存标志为假
            {
                ++collisionChecks_; // 碰撞检测计数
                collision_free = si_->checkMotion(yMin->getState(), x->getState()); // 碰撞检测
            }

            if (collision_free) // 如果无碰撞
            {
                // Add edge from yMin to x 连接x到最优父节点
                x->setParent(yMin); // 设定x父节点为最优父节点 x->parent_ = ymin;
                x->setCost(cMin);  // 设定x过最佳父节点的最小cost值 x->cost_ = cMin;
                x->setHeuristicCost(opt_->motionCostHeuristic(x->getState(), goalState_)); // 设定启发cost值
                yMin->getChildren().push_back(x); // 将x加入最佳父节点的子节点集合

                // Add x to Open
                Open_new.push_back(x); // 将x放入状态点容器中，该容器是临时存放要放入open集合中的点
                // Remove x from Unvisited
                x->setSetType(Motion::SET_CLOSED); // 设定x属性为closed
            }

//                  //modify begin 实现重新选择父节点parent
////            if (yMin->getParent() != nullptr)
//            if (yMin->getParent() != nullptr && collision_free)
//            {
//                 base::Cost adist = opt_->motionCost(x->getState(), yMin->getParent()->getState());
//                 base::Cost acNew = opt_->combineCosts(yMin->getParent()->getCost(), adist);
//                if(opt_->isCostBetterThan(acNew, x->getCost()) && si_->checkMotion(yMin->getParent()->getState(), x->getState()))
//                {
//                    // Remove this node from its parent list
//                    removeFromParent(x);//

//                    // Add this node to the new parent
//                    x->setParent(yMin->getParent());
//                    x->setCost(acNew);
//    //                x->setHeuristicCost(opt_->motionCostHeuristic(x->getState(), goalState_));
//                    x->getParent()->getChildren().push_back(x);
//    //                std::cout << "第" << i << "次xacost: " << x->getCost() << std::endl;//test code
//                }

//            }
//                    //modify end

        }  // An optimal connection from Open to x was found

//        std::cout << "xnear.size(): " << xNear.size()
//                  << " ynear.size(): " << yNear.size() << std:: endl;//test code

//        重新布线begin
//         if((lastGoalMotion_ != nullptr) && (x->getSetType() == Motion::SET_CLOSED))
        if (x->getSetType() == Motion::SET_CLOSED) // 优化解
        {
            std::vector<Motion *> hNear; // 令x点附近的Open属性并且不与x同父节点的点为h集合
            hNear.reserve(xNeighborhoodSize); // 开辟内存
            std::vector<base::Cost> costs; // h集合中每点重新布线，以x为父节点后的cost
            std::vector<base::Cost> incCosts; // x到h点的cost
            std::vector<std::size_t> sortedCostIndices; // 将重布线后的h各点按照cost从小到大排序后对应的索引值，如原先索引值1、2、3对应motion1、motion2、motion3,排序后索引1、2、3读应motion2、3、1
            CostIndexCompare compareFn(costs, *opt_); // 按照cost值比较大小，用于motion以cots值重排序

            for (unsigned int i = 0; i < xNeighborhoodSize; ++i) // 遍历x附近所有点
            {
                if ( (xNeighborhood[i]->getSetType() == Motion::SET_OPEN)
                     && ( xNeighborhood[i]->getParent() != x->getParent() ) ) // 从中找到属性为open的点并且与x不是同一个父节点的点
                {
                    hNear.push_back(xNeighborhood[i]);
                }
            }
            if (costs.size() < hNear.size()) // 开辟内存
            {
                costs.resize(hNear.size());
                incCosts.resize(hNear.size());
                sortedCostIndices.resize(hNear.size());
            }
            for (unsigned int i = 0; i < hNear.size(); ++i) // 遍历h集合，计算以x为父节点的h节点cost值
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
                 i != sortedCostIndices.begin() + hNear.size(); ++i) // 遍历索引，重布线cost从小到达依次遍历h
            {
                if (opt_->isCostBetterThan(costs[*i], hNear[*i]->getCost())) // 如果重布线后h的cost值好于原h的cost值，则实施碰撞检测和实际重布线
                {
                    Motion *hhNear = hNear[*i]; // 实际重布线hh
                    bool collision_free = false;
                    if (cacheCC_) // 如果碰撞检测缓存标志为真
                    {
                        if (!x->alreadyCC(hhNear)) // 如果x与其最优父节点没进行过碰撞检测
                        {
                            collision_free = si_->checkMotion(hhNear->getState(), x->getState()); // 进行碰撞检测
                            ++collisionChecks_; // 碰撞次数计数
                            // Due to FMT* design, it is only necessary to save unsuccesful
                            // connection attemps because of collision
                            if (!collision_free) // 如果x与其最优open父节点有碰撞
                                x->addCC(hhNear); // 缓存碰撞失败的连接
                        }
                    }
                    else // 碰撞检测缓存标志为假
                    {
                        ++collisionChecks_; // 碰撞检测计数
                        collision_free = si_->checkMotion(hhNear->getState(), x->getState()); // 碰撞检测
                    }

                    if (collision_free) // 如果无碰撞
                    {  // 将实际重布线的hh点从hh父节点的字节点容器中删除hh节点
                        std::vector<Motion*>::iterator iter = std::find(hhNear->getParent()->getChildren().begin()
                                                                        ,hhNear->getParent()->getChildren().end()
                                                                        ,hhNear);
                        //vector< int >::iterator iter=std::find(v.begin(),v.end(),num_to_find); //返回的是一个迭代器指针
                        if(iter != hhNear->getParent()->getChildren().end())//如果能找到hhNear，则在hhNear->getParent()->getChildren()里删除hhNear
                        {
                            hhNear->getParent()->getChildren().erase(iter);
                        }
                        hhNear->setParent(x); // 设定x父节点为最优父节点
                        hhNear->setCost(costs[*i]); // 设定重布线后cost值
                        x->getChildren().push_back(hhNear); // 将h加入x父节点的子节点集合
                        updateChildCosts(hhNear); // 修改hh节点之后所有字节点及其衍生字节点的cost值
                    }

//                    //modify begin 实现重新布线的父节点parent
////                     if (x->getParent() != nullptr)
//                    if (x->getParent() != nullptr && collision_free)
//                    {
//                        base::Cost nbhaIncCost;

//                         nbhaIncCost = opt_->motionCost(x->getParent()->getState(), hhNear->getState());
//                         base::Cost nbhaNewCost = opt_->combineCosts(x->getParent()->getCost(), nbhaIncCost);
//                        if(opt_->isCostBetterThan(nbhaNewCost, hhNear->getCost()) && si_->checkMotion(x->getParent()->getState(), hhNear->getState()))
//                        {
//                            // Remove this node from its parent list
//                            removeFromParent(hhNear);//

//                            // Add this node to the new parent
//                            hhNear->setParent(x->getParent());
//                            hhNear->getinc_Cost() = nbhaIncCost;
//                            hhNear->getCost() = nbhaNewCost;
//                            hhNear->getParent()->getChildren().push_back(hhNear);

//                            // Update the costs of the node's children
//                            updateChildCosts(hhNear);//

//                        }
//                    }
//                    // std::cout << "hnear.size(): " << hNear.size() << std:: endl;//test code
//                                        //modify end

                }
            }
        }
         //        重新布线end

    }      // For each node near z and in set Unvisited, try to connect it to set Open

    // Update Open
    Open_.pop(); // 删除open集合中cost值最小的状态点，该点即为该z点
    (*z)->setSetType(Motion::SET_CLOSED); // 将从open中删除的z点属性改为colsed

    // Add the nodes in Open_new to Open
    unsigned int openNewSize = Open_new.size(); // 计算z点附近无碰撞能连到树上的x点数量
    for (unsigned int i = 0; i < openNewSize; ++i) // 将Open_new中所有点放入open集合中，并设定为open属性
    {
        Open_.insert(Open_new[i]);
        Open_new[i]->setSetType(Motion::SET_OPEN);
    }
    Open_new.clear(); //清空临时open容器

    if (Open_.empty()) // 如果open集合空，即在所有树上点都无法以当前搜索半径搜索到新的未访问点，返回false
    {
//        if (lastGoalMotion_ == nullptr)
//        {
//            OMPL_INFORM("Open is empty before path was found --> no feasible path exists");
//        }
        if (lastGoalMotion_ != nullptr)
        {
            OMPL_INFORM("Open_set is empty");
        }
        return false;
    }

    // Take the top of Open as the new z
    *z = Open_.top()->data; // 将open集合中cost最小的点赋值给z点，返回true

    return true;
}


bool ompl::geometric::FMT::expandTreeFromNode(Motion **z)
{
    // Find all nodes that are near z, and also in set Unvisited 寻找z附近所有未访问过的点
    std::vector<Motion *> xNear; // z附近x个状态点，不包含z本身
    const std::vector<Motion *> &zNeighborhood = neighborhoods_[*z]; // z附近的状态点，不包含z本身
    const unsigned int zNeighborhoodSize = zNeighborhood.size(); // z附近状态点个数
    xNear.reserve(zNeighborhoodSize); // 开辟xNear内存空间，大小是z附近状态点个数的的大小

    for (unsigned int i = 0; i < zNeighborhoodSize; ++i) // 遍历z附近所有点，将所有未访问过点放一起
    {
        Motion *x = zNeighborhood[i]; // z附近点中的一个赋值给x
        if (x->getSetType() == Motion::SET_UNVISITED) // 如果该附近点未被访问过
        {
            saveNeighborhood(x); // 获取该点的附近点
            if (nearestK_)
            {
                // Only include neighbors that are mutually k-nearest
                // Relies on NN datastructure returning k-nearest in sorted order
                const base::Cost connCost = opt_->motionCost((*z)->getState(), x->getState());
                const base::Cost worstCost = opt_->motionCost(neighborhoods_[x].back()->getState(), x->getState());

                if (opt_->isCostBetterThan(worstCost, connCost))
                    continue;
                xNear.push_back(x);
            }
            else
                xNear.push_back(x); // 将该点放入xNear，xNear中存的都是z附近未访问过的点
        }
    }

    // For each node near z and in set Unvisited, attempt to connect it to set Open 尝试对z附近未访问过的点放入open集合
    std::vector<Motion *> yNear; // z附近未访问的一个点x的属于open集合的附近点
    std::vector<Motion *> Open_new; // 新加入open集合中的几个点
    const unsigned int xNearSize = xNear.size(); // z附近未访问点数量
    for (unsigned int i = 0; i < xNearSize; ++i) // 遍历z附近未访问点
    {
        Motion *x = xNear[i]; // z附近未访问点中的一个点x

        // Find all nodes that are near x and in set Open 该未访问点x附近所有open集合中的点
        const std::vector<Motion *> &xNeighborhood = neighborhoods_[x]; // 未访问点x附近所有点

        const unsigned int xNeighborhoodSize = xNeighborhood.size(); // x附近点个数
        yNear.reserve(xNeighborhoodSize); // 开辟内存
        for (unsigned int j = 0; j < xNeighborhoodSize; ++j) // 遍历z附近未访问点x附近的所有点
        {
            if (xNeighborhood[j]->getSetType() == Motion::SET_OPEN) // 从中找到属性为open的点
                yNear.push_back(xNeighborhood[j]); // 将x附近的属于open集合的点放到一起
        }

        // Find the lowest cost-to-come connection from Open to x 寻找x附近open集合点到x点的最小cost连接
        base::Cost cMin(std::numeric_limits<double>::infinity()); // cost最小值变量，初始为无穷大
        Motion *yMin = getBestParent(x, yNear, cMin); // 在x附近属性为open的点集合找到x的最佳父状态点，并得到x过最佳父节点的最小cost值
        yNear.clear(); // 清空x附近属性为open的点集合
        //3.15 2021 modify
        bool expand = true;
//        if(lastGoalMotion_!=nullptr)
//        {
//        base::Cost improvedsolutionHeuristic ;
//        const base::Cost costToGo =
//                opt_->costToGo(x->getState(), pdef_->getGoal().get());
//        improvedsolutionHeuristic = opt_->combineCosts(cMin,costToGo);
////        if(opt_->isCostBetterThan(improvedsolutionHeuristic,lastGoalMotion_->getCost()))
////        //3.15 2021 modify  这里将cost（ymin）+cost to go > lastcost的点排除
////        base::Cost improvedsolutionHeuristic;
////        const base::Cost costToGo =
////                opt_->costToGo(yMin->getState(), pdef_->getGoal().get());
////        improvedsolutionHeuristic = opt_->combineCosts(yMin->getCost(),costToGo);
////        //3.15 2021 modify
//        if(!opt_->isCostBetterThan(improvedsolutionHeuristic,lastGoalMotion_->getCost()))
//            expand = false;
//        }

        // If an optimal connection from Open to x was found 如果x到附近open集合的最佳父状态点存在
        if((yMin != nullptr)&&(expand==true))// 如果x最优父状态点不空
        {
//            std::cout << "lastgoalmotioncost: "<< lastGoalMotion_->getCost()<< std::endl;
            bool collision_free = false;
            if (cacheCC_) // 如果碰撞检测缓存标志为真，碰撞检测缓存用于减少碰撞检测，即曾经检测过的两个点不再检测
            {
                if (!yMin->alreadyCC(x)) // 如果x与其最优父节点没进行过碰撞检测
                {
                    collision_free = si_->checkMotion(yMin->getState(), x->getState()); // 进行碰撞检测
                    ++collisionChecks_; // 碰撞次数计数
                    // Due to FMT* design, it is only necessary to save unsuccesful
                    // connection attemps because of collision
                    if (!collision_free) // 如果x与其最优open父节点有碰撞
                        yMin->addCC(x); // 缓存碰撞失败的连接
                }
            }
            else // 碰撞检测缓存标志为假
            {
                ++collisionChecks_; // 碰撞检测计数
                collision_free = si_->checkMotion(yMin->getState(), x->getState()); // 碰撞检测
            }

            if (collision_free) // 如果无碰撞
            {
                // Add edge from yMin to x 连接x到最优父节点
                x->setParent(yMin); // 设定x父节点为最优父节点
                x->setCost(cMin);  // 设定x过最佳父节点的最小cost值
                x->setHeuristicCost(opt_->motionCostHeuristic(x->getState(), goalState_)); // 设定启发cost值
                yMin->getChildren().push_back(x); // 将x加入最佳父节点的子节点集合

                // Add x to Open
                Open_new.push_back(x); // 将x放入状态点容器中，该容器是临时存放要放入open集合中的点
                // Remove x from Unvisited
                x->setSetType(Motion::SET_CLOSED); // 设定x属性为closed
            }
        }  // An optimal connection from Open to x was found

        if ( (lastGoalMotion_ != nullptr) && (x->getSetType() == Motion::SET_CLOSED) ) // 优化解
        {
            std::vector<Motion *> hNear; // 令x点附近的Open属性并且不与x同父节点的点为h集合
            hNear.reserve(xNeighborhoodSize); // 开辟内存
            std::vector<base::Cost> costs; // h集合中每点重新布线，以x为父节点后的cost
            std::vector<base::Cost> incCosts; // x到h点的cost
            std::vector<std::size_t> sortedCostIndices; // 将重布线后的h各点按照cost从小到大排序后对应的索引值，如原先索引值1、2、3对应motion1、motion2、motion3,排序后索引1、2、3读应motion2、3、1
            CostIndexCompare compareFn(costs, *opt_); // 按照cost值比较大小，用于motion以cots值重排序

            for (unsigned int i = 0; i < xNeighborhoodSize; ++i) // 遍历x附近所有点
            {
                if ( (xNeighborhood[i]->getSetType() == Motion::SET_OPEN)
                     && ( xNeighborhood[i]->getParent() != x->getParent() ) ) // 从中找到属性为open的点并且与x不是同一个父节点的点
                {
                    hNear.push_back(xNeighborhood[i]);
                }
            }
            if (costs.size() < hNear.size()) // 开辟内存
            {
                costs.resize(hNear.size());
                incCosts.resize(hNear.size());
                sortedCostIndices.resize(hNear.size());
            }
            for (unsigned int i = 0; i < hNear.size(); ++i) // 遍历h集合，计算以x为父节点的h节点cost值
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
                 i != sortedCostIndices.begin() + hNear.size(); ++i) // 遍历索引，重布线cost从小到达依次遍历h
            {
                if (opt_->isCostBetterThan(costs[*i], hNear[*i]->getCost())) // 如果重布线后h的cost值好于原h的cost值，则实施碰撞检测和实际重布线
                {
                    Motion *hhNear = hNear[*i]; // 实际重布线hh
                    bool collision_free = false;
                    if (cacheCC_) // 如果碰撞检测缓存标志为真
                    {
                        if (!x->alreadyCC(hhNear)) // 如果x与其最优父节点没进行过碰撞检测
                        {
                            collision_free = si_->checkMotion(hhNear->getState(), x->getState()); // 进行碰撞检测
                            ++collisionChecks_; // 碰撞次数计数
                            // Due to FMT* design, it is only necessary to save unsuccesful
                            // connection attemps because of collision
                            if (!collision_free) // 如果x与其最优open父节点有碰撞
                                x->addCC(hhNear); // 缓存碰撞失败的连接
                        }
                    }
                    else // 碰撞检测缓存标志为假
                    {
                        ++collisionChecks_; // 碰撞检测计数
                        collision_free = si_->checkMotion(hhNear->getState(), x->getState()); // 碰撞检测
                    }

                    if (collision_free) // 如果无碰撞
                    {  // 将实际重布线的hh点从hh父节点的字节点容器中删除hh节点
                        std::vector<Motion*>::iterator iter = std::find(hhNear->getParent()->getChildren().begin()
                                                                        ,hhNear->getParent()->getChildren().end()
                                                                        ,hhNear);
                        if(iter != hhNear->getParent()->getChildren().end())
                        {
                            hhNear->getParent()->getChildren().erase(iter);
                        }
                        hhNear->setParent(x); // 设定x父节点为最优父节点
                        hhNear->setCost(costs[*i]); // 设定重布线后cost值
                        x->getChildren().push_back(hhNear); // 将h加入x父节点的子节点集合
                        updateChildCosts(hhNear); // 修改hh节点之后所有字节点及其衍生字节点的cost值
                    }
                }
            }
        }
    }      // For each node near z and in set Unvisited, try to connect it to set Open

    // Update Open
    Open_.pop(); // 删除open集合中cost值最小的状态点，该点即使该z点
    (*z)->setSetType(Motion::SET_CLOSED); // 将从open中删除的z点属性改为colsed

    // Add the nodes in Open_new to Open
    unsigned int openNewSize = Open_new.size(); // 计算z点附近无碰撞能连到树上的x点数量
    for (unsigned int i = 0; i < openNewSize; ++i) // 将Open_new中所有点放入open集合中，并设定为open属性
    {
        Open_.insert(Open_new[i]);
        Open_new[i]->setSetType(Motion::SET_OPEN);
    }
    Open_new.clear(); //清空临时open容器

    if (Open_.empty()) // 如果open集合空，即在所有树上点都无法以当前搜索半径搜索到新的未访问点，返回false
    {
        if (lastGoalMotion_ != nullptr)
        {
//            OMPL_INFORM("Open is empty before path was found --> no feasible path exists");
//        }
//        else {
            OMPL_INFORM("Open_set is empty");
        }
        return false;
    }

    // Take the top of Open as the new z
    *z = Open_.top()->data; // 将open集合中cost最小的点赋值给z点，返回true

    return true;
}

ompl::geometric::FMT::Motion *ompl::geometric::FMT::getBestParent(Motion *m, std::vector<Motion *> &neighbors,
                                                                  base::Cost &cMin)
{
    Motion *min = nullptr;
    const unsigned int neighborsSize = neighbors.size();
    for (unsigned int j = 0; j < neighborsSize; ++j)
    {
        const base::State *s = neighbors[j]->getState();
        const base::Cost dist = opt_->motionCost(s, m->getState());
        const base::Cost cNew = opt_->combineCosts(neighbors[j]->getCost(), dist);

        if (opt_->isCostBetterThan(cNew, cMin))
        {
            min = neighbors[j];
            cMin = cNew;
        }
    }
    return min;
}

void ompl::geometric::FMT::updateChildCosts(Motion *m)
{
    for (unsigned int i = 0; i < m->getChildren().size(); ++i)
    {
        base::Cost incCost;
        incCost = opt_->motionCost(m->getState(), m->getChildren()[i]->getState());
        m->getChildren()[i]->setCost(opt_->combineCosts(m->getCost(), incCost));
        updateChildCosts(m->getChildren()[i]); // 递归调用，将m的所有子节点及其子节点的子节点的代价值进行更新
    }
}

void ompl::geometric::FMT::updateNeighborhood(Motion *m, const std::vector<Motion *> nbh)
{
    for (auto i : nbh)
    {
        // If CLOSED, the neighborhood already exists. If neighborhood already exists, we have
        // to insert the node in the corresponding place of the neighborhood of the neighbor of m.
        if (i->getSetType() == Motion::SET_CLOSED || neighborhoods_.find(i) != neighborhoods_.end())
        {
            const base::Cost connCost = opt_->motionCost(i->getState(), m->getState());
            const base::Cost worstCost = opt_->motionCost(neighborhoods_[i].back()->getState(), i->getState());

            if (opt_->isCostBetterThan(worstCost, connCost))
                continue;

            // Insert the neighbor in the vector in the correct order
            std::vector<Motion *> &nbhToUpdate = neighborhoods_[i];
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
        else
        {
            std::vector<Motion *> nbh2;
            if (nearestK_)
                nn_->nearestK(m, NNk_, nbh2);
            else
                nn_->nearestR(m, NNr_, nbh2);

            if (!nbh2.empty())
            {
                // Save the neighborhood but skip the first element, since it will be motion m
                neighborhoods_[i] = std::vector<Motion *>(nbh2.size() - 1, nullptr);
                std::copy(nbh2.begin() + 1, nbh2.end(), neighborhoods_[i].begin());
            }
            else
            {
                // Save an empty neighborhood
                neighborhoods_[i] = std::vector<Motion *>(0);
            }
        }
    }
}
