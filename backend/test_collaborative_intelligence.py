#!/usr/bin/env python3
"""
Test Suite for Collaborative Intelligence Services

Tests the extracted collaborative intelligence services including peer learning,
group formation, collective intelligence, and social networks.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import collaborative intelligence services
from quantum_intelligence.services.collaborative import (
    PeerLearningOptimizer,
    PeerMatchingNetwork,
    PeerCompatibilityAnalyzer,
    PeerTutoringSystem,
    GroupFormationEngine,
    CollectiveIntelligenceHarvester,
    WisdomAggregationEngine,
    GroupProblemSolvingAnalyzer,
    SocialLearningNetworkAnalyzer,
    CollaborativeIntelligenceEngine,
    CollaborativeLearningOrchestrator
)


class CollaborativeIntelligenceTestSuite:
    """Comprehensive test suite for collaborative intelligence services"""
    
    def __init__(self):
        self.test_results = []
        self.sample_participants = self._create_sample_participants()
        self.sample_learning_objectives = [
            "Master collaborative problem solving",
            "Develop effective communication skills",
            "Learn advanced data analysis techniques"
        ]
    
    def _create_sample_participants(self):
        """Create sample participant data for testing"""
        return [
            {
                'user_id': 'user_001',
                'name': 'Alice Johnson',
                'learning_style': {'visual': 0.8, 'auditory': 0.3, 'kinesthetic': 0.5, 'reading_writing': 0.7},
                'personality': {'openness': 0.8, 'conscientiousness': 0.7, 'extraversion': 0.6, 'agreeableness': 0.8, 'neuroticism': 0.3},
                'skills': {'python': 0.8, 'data_analysis': 0.7, 'machine_learning': 0.6},
                'learning_goals': ['machine_learning', 'data_science'],
                'availability': {'09:00': True, '10:00': True, '14:00': True, '15:00': True},
                'background': {'field': 'computer_science', 'experience_level': 'intermediate'},
                'activity_level': 0.8,
                'expertise_areas': ['programming', 'data_analysis']
            },
            {
                'user_id': 'user_002',
                'name': 'Bob Smith',
                'learning_style': {'visual': 0.4, 'auditory': 0.8, 'kinesthetic': 0.6, 'reading_writing': 0.5},
                'personality': {'openness': 0.7, 'conscientiousness': 0.8, 'extraversion': 0.8, 'agreeableness': 0.7, 'neuroticism': 0.2},
                'skills': {'python': 0.5, 'data_analysis': 0.9, 'statistics': 0.8},
                'learning_goals': ['advanced_statistics', 'data_science'],
                'availability': {'09:00': True, '10:00': True, '11:00': True, '14:00': True},
                'background': {'field': 'statistics', 'experience_level': 'advanced'},
                'activity_level': 0.9,
                'expertise_areas': ['statistics', 'data_analysis']
            },
            {
                'user_id': 'user_003',
                'name': 'Carol Davis',
                'learning_style': {'visual': 0.6, 'auditory': 0.5, 'kinesthetic': 0.8, 'reading_writing': 0.4},
                'personality': {'openness': 0.9, 'conscientiousness': 0.6, 'extraversion': 0.4, 'agreeableness': 0.9, 'neuroticism': 0.4},
                'skills': {'python': 0.3, 'data_analysis': 0.4, 'domain_expertise': 0.9},
                'learning_goals': ['programming', 'data_science'],
                'availability': {'09:00': True, '10:00': True, '14:00': True, '15:00': True},  # Better overlap with user_002
                'background': {'field': 'biology', 'experience_level': 'beginner'},
                'activity_level': 0.6,
                'expertise_areas': ['domain_knowledge', 'research']
            },
            {
                'user_id': 'user_004',
                'name': 'David Wilson',
                'learning_style': {'visual': 0.7, 'auditory': 0.6, 'kinesthetic': 0.4, 'reading_writing': 0.8},
                'personality': {'openness': 0.6, 'conscientiousness': 0.9, 'extraversion': 0.7, 'agreeableness': 0.6, 'neuroticism': 0.3},
                'skills': {'python': 0.7, 'data_analysis': 0.6, 'project_management': 0.8},
                'learning_goals': ['leadership', 'data_science'],
                'availability': {'09:00': True, '14:00': True, '15:00': True, '16:00': True},
                'background': {'field': 'business', 'experience_level': 'intermediate'},
                'activity_level': 0.7,
                'expertise_areas': ['project_management', 'business_analysis']
            }
        ]
    
    async def run_all_tests(self):
        """Run all collaborative intelligence tests"""
        print("🧠 Starting Collaborative Intelligence Services Test Suite")
        print("=" * 70)
        
        # Test 1: Peer Learning Services
        await self.test_peer_learning_services()
        
        # Test 2: Group Formation Services
        await self.test_group_formation_services()
        
        # Test 3: Collective Intelligence Services
        await self.test_collective_intelligence_services()
        
        # Test 4: Social Networks Services
        await self.test_social_networks_services()
        
        # Test 5: Collaborative Intelligence Engine
        await self.test_collaborative_intelligence_engine()
        
        # Test 6: End-to-End Collaborative Learning
        await self.test_end_to_end_collaborative_learning()
        
        # Print results
        self.print_test_results()
    
    async def test_peer_learning_services(self):
        """Test peer learning optimization services"""
        print("\n1. Testing Peer Learning Services...")

        try:
            # Test peer compatibility analyzer
            compatibility_analyzer = PeerCompatibilityAnalyzer()
            compatibility_result = await compatibility_analyzer.analyze_peer_compatibility(
                self.sample_participants[0], self.sample_participants[1]
            )

            assert compatibility_result['status'] == 'success', f"Compatibility analysis failed: {compatibility_result}"
            assert 'compatibility_analysis' in compatibility_result
            assert 'overall_compatibility' in compatibility_result['compatibility_analysis']

            # Test peer matching network
            peer_matcher = PeerMatchingNetwork(compatibility_analyzer)
            matching_result = await peer_matcher.find_optimal_peers(
                self.sample_participants[0], self.sample_participants[1:], {}
            )

            assert matching_result['status'] == 'success', f"Peer matching failed: {matching_result}"
            assert 'ranked_matches' in matching_result

            # Test peer tutoring system
            tutoring_system = PeerTutoringSystem(peer_matcher)
            tutoring_result = await tutoring_system.create_tutoring_session(
                self.sample_participants[1], self.sample_participants[2], 'data_analysis'
            )

            assert tutoring_result['status'] == 'success', f"Tutoring session creation failed: {tutoring_result}"
            assert 'session_id' in tutoring_result

            # Test peer learning optimizer
            peer_optimizer = PeerLearningOptimizer()
            optimization_result = await peer_optimizer.optimize_peer_learning_experience(
                self.sample_participants, self.sample_learning_objectives
            )

            assert optimization_result['status'] == 'success', f"Peer optimization failed: {optimization_result}"
            assert 'optimal_groups' in optimization_result

            self.test_results.append(("Peer Learning Services", "PASS", "All peer learning components working"))

        except Exception as e:
            import traceback
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            self.test_results.append(("Peer Learning Services", "FAIL", error_details))
    
    async def test_group_formation_services(self):
        """Test group formation services"""
        print("\n2. Testing Group Formation Services...")
        
        try:
            # Test group formation engine
            group_formation_engine = GroupFormationEngine()
            
            formation_criteria = {
                'strategy': 'compatibility_based',
                'learning_objectives': self.sample_learning_objectives
            }
            
            formation_result = await group_formation_engine.form_learning_groups(
                self.sample_participants, formation_criteria
            )
            
            assert formation_result['status'] == 'success'
            assert 'groups' in formation_result
            assert len(formation_result['groups']) > 0
            
            # Test different formation strategies
            strategies = ['diversity_based', 'skill_balanced', 'hybrid_optimized']
            
            for strategy in strategies:
                formation_criteria['strategy'] = strategy
                strategy_result = await group_formation_engine.form_learning_groups(
                    self.sample_participants, formation_criteria
                )
                assert strategy_result['status'] == 'success'
            
            self.test_results.append(("Group Formation Services", "PASS", "All group formation strategies working"))
            
        except Exception as e:
            self.test_results.append(("Group Formation Services", "FAIL", str(e)))
    
    async def test_collective_intelligence_services(self):
        """Test collective intelligence services"""
        print("\n3. Testing Collective Intelligence Services...")
        
        try:
            # Create sample contributions
            sample_contributions = [
                {
                    'contributor_id': 'user_001',
                    'content': 'Machine learning requires understanding of statistics and programming',
                    'contribution_type': 'knowledge_sharing',
                    'concepts': {'machine_learning': 0.8, 'statistics': 0.7},
                    'insights': ['ML combines multiple disciplines'],
                    'evidence': ['research_paper_1', 'practical_experience']
                },
                {
                    'contributor_id': 'user_002',
                    'content': 'Statistical analysis is fundamental for data science applications',
                    'contribution_type': 'expertise_sharing',
                    'concepts': {'statistics': 0.9, 'data_science': 0.8},
                    'insights': ['Statistics provides foundation for data science'],
                    'evidence': ['academic_background', 'industry_experience']
                },
                {
                    'contributor_id': 'user_003',
                    'content': 'Domain knowledge is crucial for effective data analysis',
                    'contribution_type': 'practical_insight',
                    'concepts': {'domain_expertise': 0.9, 'data_analysis': 0.6},
                    'insights': ['Domain context improves analysis quality'],
                    'evidence': ['field_experience', 'case_studies']
                }
            ]
            
            # Test collective intelligence harvester
            intelligence_harvester = CollectiveIntelligenceHarvester()
            harvesting_result = await intelligence_harvester.harvest_collective_intelligence(
                'data_science_fundamentals', sample_contributions
            )
            
            assert harvesting_result['status'] == 'success'
            assert 'collective_wisdom' in harvesting_result
            
            # Test wisdom aggregation engine
            wisdom_aggregator = WisdomAggregationEngine(intelligence_harvester)
            aggregation_result = await wisdom_aggregator.aggregate_wisdom_across_topics(
                ['data_science_fundamentals']
            )
            
            assert aggregation_result['status'] == 'success'
            assert 'aggregated_wisdom' in aggregation_result
            
            self.test_results.append(("Collective Intelligence Services", "PASS", "Intelligence harvesting and aggregation working"))
            
        except Exception as e:
            self.test_results.append(("Collective Intelligence Services", "FAIL", str(e)))
    
    async def test_social_networks_services(self):
        """Test social networks services"""
        print("\n4. Testing Social Networks Services...")
        
        try:
            # Create sample network data
            network_data = {
                'network_id': 'test_network',
                'network_name': 'Test Learning Network',
                'nodes': self.sample_participants,
                'edges': [
                    {'source': 'user_001', 'target': 'user_002', 'weight': 0.8, 'knowledge_exchange': True},
                    {'source': 'user_002', 'target': 'user_003', 'weight': 0.6, 'knowledge_exchange': True},
                    {'source': 'user_003', 'target': 'user_004', 'weight': 0.7, 'knowledge_exchange': False},
                    {'source': 'user_001', 'target': 'user_004', 'weight': 0.5, 'knowledge_exchange': True}
                ]
            }
            
            # Test social learning network analyzer
            network_analyzer = SocialLearningNetworkAnalyzer()
            network_result = await network_analyzer.analyze_social_learning_network(network_data)
            
            assert network_result['status'] == 'success'
            assert 'social_network' in network_result
            assert 'network_metrics' in network_result['social_network']
            
            self.test_results.append(("Social Networks Services", "PASS", "Social network analysis working"))
            
        except Exception as e:
            self.test_results.append(("Social Networks Services", "FAIL", str(e)))
    
    async def test_collaborative_intelligence_engine(self):
        """Test collaborative intelligence engine"""
        print("\n5. Testing Collaborative Intelligence Engine...")
        
        try:
            # Test collaborative intelligence engine
            collaborative_engine = CollaborativeIntelligenceEngine()
            
            collaboration_config = {
                'type': 'group_projects',
                'group_formation_strategy': 'compatibility_based',
                'constraints': {'group_size': 2}
            }
            
            orchestration_result = await collaborative_engine.orchestrate_collaborative_learning(
                self.sample_participants, self.sample_learning_objectives, collaboration_config
            )
            
            assert orchestration_result['status'] == 'success'
            assert 'session_id' in orchestration_result
            assert 'collaborative_session' in orchestration_result
            
            # Test session monitoring
            session_id = orchestration_result['session_id']
            real_time_data = {
                'collaboration_score': 0.8,
                'overall_engagement': 0.7,
                'learning_progress': 0.6
            }
            
            monitoring_result = await collaborative_engine.monitor_collaborative_session(
                session_id, real_time_data
            )
            
            assert monitoring_result['status'] == 'success'
            assert 'collaboration_dynamics' in monitoring_result
            
            self.test_results.append(("Collaborative Intelligence Engine", "PASS", "Engine orchestration and monitoring working"))
            
        except Exception as e:
            self.test_results.append(("Collaborative Intelligence Engine", "FAIL", str(e)))
    
    async def test_end_to_end_collaborative_learning(self):
        """Test end-to-end collaborative learning workflow"""
        print("\n6. Testing End-to-End Collaborative Learning...")
        
        try:
            # Test collaborative learning orchestrator
            collaborative_engine = CollaborativeIntelligenceEngine()
            orchestrator = CollaborativeLearningOrchestrator(collaborative_engine)
            
            # Test quick peer session
            peer_session_result = await orchestrator.quick_peer_session(
                self.sample_participants[0], self.sample_participants[1], 
                'collaborative problem solving'
            )
            
            assert peer_session_result['status'] == 'success'
            
            # Test group project session
            project_config = {'group_size': 4, 'duration_minutes': 90}
            group_session_result = await orchestrator.group_project_session(
                self.sample_participants, self.sample_learning_objectives, project_config
            )
            
            assert group_session_result['status'] == 'success'
            
            # Test knowledge sharing circle
            knowledge_domains = ['data_science', 'machine_learning', 'statistics']
            sharing_result = await orchestrator.knowledge_sharing_circle(
                self.sample_participants, knowledge_domains
            )
            
            assert sharing_result['status'] == 'success'
            
            self.test_results.append(("End-to-End Collaborative Learning", "PASS", "Complete workflow functioning"))
            
        except Exception as e:
            self.test_results.append(("End-to-End Collaborative Learning", "FAIL", str(e)))
    
    def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "=" * 70)
        print("🧠 COLLABORATIVE INTELLIGENCE SERVICES TEST RESULTS")
        print("=" * 70)
        
        passed = 0
        failed = 0
        
        for test_name, status, details in self.test_results:
            status_icon = "✅" if status == "PASS" else "❌"
            print(f"{status_icon} {test_name}: {status}")
            if status == "FAIL":
                print(f"   Error: {details}")
                failed += 1
            else:
                print(f"   Details: {details}")
                passed += 1
        
        print("\n" + "=" * 70)
        print(f"📊 SUMMARY: {passed} passed, {failed} failed out of {len(self.test_results)} tests")
        
        if failed == 0:
            print("🎉 ALL COLLABORATIVE INTELLIGENCE SERVICES TESTS PASSED!")
            print("\n✨ Collaborative Intelligence Services are ready for:")
            print("   • Advanced peer learning optimization")
            print("   • Intelligent group formation algorithms")
            print("   • Collective intelligence harvesting")
            print("   • Social learning network analysis")
            print("   • End-to-end collaborative orchestration")
        else:
            print(f"⚠️  {failed} test(s) failed. Please review the errors above.")
        
        print("=" * 70)


async def main():
    """Run the collaborative intelligence test suite"""
    test_suite = CollaborativeIntelligenceTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
