"""
Simplified Materials Decision Agent Demo
Works without LangChain dependencies for demonstration purposes
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional

class SimpleMaterialsAgent:
    """Simplified version of the materials decision agent for demo purposes"""
    
    def __init__(self):
        self.material_types = [
            "slide", "infographic", "chart", "video_explainer", 
            "social_media_post", "presentation_deck"
        ]
        
        self.priority_levels = ["high", "medium", "low"]
    
    def analyze_claims_for_materials(self, verified_claims: List[Dict], client_context: str) -> List[Dict]:
        """Generate material recommendations based on verified claims"""
        
        recommendations = []
        
        for i, claim in enumerate(verified_claims):
            # Simple rule-based material selection
            claim_text = claim.get('claim', '').lower()
            
            if 'market' in claim_text and 'billion' in claim_text:
                # Market size claims → Infographic
                rec = {
                    "material_id": str(uuid.uuid4()),
                    "material_type": "infographic",
                    "title": f"Singapore E-commerce Market Size 2023",
                    "description": "Visual representation of market size with key statistics",
                    "claim_references": [claim.get('claim_id', f'claim_{i+1}')],
                    "content_requirements": {
                        "style": "professional",
                        "data_visualization": "required",
                        "text_amount": "minimal",
                        "color_scheme": "corporate",
                        "special_elements": ["charts", "statistics", "icons"]
                    },
                    "priority": "high",
                    "estimated_time_minutes": 45,
                    "reasoning": "Market size data is highly impactful for startup presentations"
                }
                recommendations.append(rec)
            
            elif 'market share' in claim_text or 'leads' in claim_text:
                # Market share claims → Chart
                rec = {
                    "material_id": str(uuid.uuid4()),
                    "material_type": "chart",
                    "title": "E-commerce Platform Market Share",
                    "description": "Pie chart showing competitive landscape",
                    "claim_references": [claim.get('claim_id', f'claim_{i+1}')],
                    "content_requirements": {
                        "style": "modern",
                        "data_visualization": "required",
                        "text_amount": "minimal",
                        "color_scheme": "vibrant",
                        "special_elements": ["charts", "percentages"]
                    },
                    "priority": "medium",
                    "estimated_time_minutes": 30,
                    "reasoning": "Market share visualization helps understand competitive position"
                }
                recommendations.append(rec)
            
            elif 'consumer' in claim_text and '%' in claim_text:
                # Consumer behavior claims → Slide
                rec = {
                    "material_id": str(uuid.uuid4()),
                    "material_type": "slide",
                    "title": "Consumer Online Shopping Behavior",
                    "description": "Key consumer statistics with visual highlights",
                    "claim_references": [claim.get('claim_id', f'claim_{i+1}')],
                    "content_requirements": {
                        "style": "professional",
                        "data_visualization": "optional",
                        "text_amount": "moderate",
                        "color_scheme": "corporate",
                        "special_elements": ["statistics", "icons"]
                    },
                    "priority": "medium",
                    "estimated_time_minutes": 25,
                    "reasoning": "Consumer behavior stats support business case"
                }
                recommendations.append(rec)
        
        # Add a comprehensive presentation deck if we have multiple claims
        if len(verified_claims) >= 2:
            deck_rec = {
                "material_id": str(uuid.uuid4()),
                "material_type": "presentation_deck",
                "title": "Singapore E-commerce Market Overview",
                "description": "Complete 6-slide presentation covering all key points",
                "claim_references": [claim.get('claim_id', f'claim_{i+1}') for i, claim in enumerate(verified_claims)],
                "content_requirements": {
                    "style": "professional",
                    "data_visualization": "required",
                    "text_amount": "detailed",
                    "color_scheme": "corporate",
                    "special_elements": ["charts", "statistics", "conclusions"]
                },
                "priority": "high",
                "estimated_time_minutes": 90,
                "reasoning": "Comprehensive deck provides complete story for client presentation"
            }
            recommendations.append(deck_rec)
        
        return recommendations
    
    def prioritize_materials(self, recommendations: List[Dict], time_limit: int = 120) -> tuple:
        """Prioritize materials based on time constraints and impact"""
        
        # Sort by priority and time
        priority_order = {"high": 3, "medium": 2, "low": 1}
        
        sorted_materials = sorted(
            recommendations,
            key=lambda x: (priority_order.get(x.get("priority", "low"), 1), -x.get("estimated_time_minutes", 0)),
            reverse=True
        )
        
        # Select materials that fit within time limit
        selected_materials = []
        total_time = 0
        
        for material in sorted_materials:
            material_time = material.get("estimated_time_minutes", 30)
            if total_time + material_time <= time_limit:
                selected_materials.append(material)
                total_time += material_time
        
        return selected_materials, total_time
    
    def create_generation_queue(self, selected_materials: List[Dict]) -> List[Dict]:
        """Create generation queue with detailed instructions"""
        
        generation_queue = []
        
        for material in selected_materials:
            task = {
                "task_id": str(uuid.uuid4()),
                "material_id": material["material_id"],
                "type": material["material_type"],
                "title": material["title"],
                "description": material["description"],
                "content_requirements": material["content_requirements"],
                "claim_references": material["claim_references"],
                "priority": material["priority"],
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "instructions": self.generate_instructions(material)
            }
            generation_queue.append(task)
        
        return generation_queue
    
    def generate_instructions(self, material: Dict) -> str:
        """Generate creation instructions for each material type"""
        
        material_type = material["material_type"]
        title = material["title"]
        content_req = material["content_requirements"]
        
        instructions = {
            "slide": f"Create a professional slide titled '{title}'. Include headline, 2-3 key statistics, and supporting visual. Use {content_req.get('color_scheme', 'corporate')} color scheme.",
            
            "infographic": f"Design an infographic for '{title}'. Combine statistics, icons, and minimal text in a visually appealing layout. Style: {content_req.get('style', 'professional')}.",
            
            "chart": f"Create a data visualization for '{title}'. Choose appropriate chart type (pie, bar, line) based on data. Include clear labels and source attribution.",
            
            "video_explainer": f"Produce a 60-second video explainer for '{title}'. Include voice-over script, simple animations, and key statistics overlay.",
            
            "presentation_deck": f"Build a complete presentation deck for '{title}'. Include title slide, agenda, key findings (3-4 slides), and conclusion with next steps.",
            
            "social_media_post": f"Create social media content for '{title}'. Design for LinkedIn/Twitter with eye-catching visual and concise copy."
        }
        
        return instructions.get(material_type, f"Create {material_type} for {title}")

def demo_simple_materials_agent():
    """Run the simplified materials agent demo"""
    
    print("🎯 SIMPLIFIED MATERIALS DECISION AGENT DEMO")
    print("=" * 70)
    
    # Sample verified claims
    verified_claims = [
        {
            "claim_id": "claim_001",
            "claim": "Singapore's e-commerce market reached SGD 9 billion in sales in 2023",
            "verdict": "TRUE",
            "confidence": 0.92,
            "evidence": [
                {
                    "source": "Singapore Department of Statistics",
                    "summary": "Official e-commerce sales data confirms SGD 9.1B in 2023"
                }
            ]
        },
        {
            "claim_id": "claim_002",
            "claim": "Shopee leads Singapore's e-commerce market with 35% market share",
            "verdict": "TRUE",
            "confidence": 0.85,
            "evidence": [
                {
                    "source": "TechCrunch Southeast Asia Report 2023",
                    "summary": "Shopee maintains market leadership position"
                }
            ]
        },
        {
            "claim_id": "claim_003",
            "claim": "Over 80% of Singapore consumers shop online at least once monthly",
            "verdict": "TRUE",
            "confidence": 0.88,
            "evidence": [
                {
                    "source": "Consumer Association of Singapore Survey 2023",
                    "summary": "83% of surveyed consumers report monthly online shopping"
                }
            ]
        }
    ]
    
    client_context = "Small e-commerce startup in Singapore looking to understand market opportunities"
    
    print(f"📋 INPUT:")
    print(f"   • Verified claims: {len(verified_claims)}")
    print(f"   • Client context: {client_context}")
    
    # Initialize agent
    agent = SimpleMaterialsAgent()
    
    # Generate recommendations
    print(f"\n🔄 ANALYZING CLAIMS...")
    recommendations = agent.analyze_claims_for_materials(verified_claims, client_context)
    print(f"   ✅ Generated {len(recommendations)} material recommendations")
    
    # Prioritize materials
    print(f"\n⚖️ PRIORITIZING MATERIALS...")
    selected_materials, total_time = agent.prioritize_materials(recommendations, time_limit=120)
    print(f"   ✅ Selected {len(selected_materials)} materials (Total: {total_time} minutes)")
    
    # Create generation queue
    print(f"\n🚀 CREATING GENERATION QUEUE...")
    generation_queue = agent.create_generation_queue(selected_materials)
    print(f"   ✅ Created {len(generation_queue)} generation tasks")
    
    # Display results
    print(f"\n📊 RESULTS:")
    print("=" * 50)
    
    print(f"\n🎨 MATERIAL RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        selected = "✅ SELECTED" if rec in selected_materials else "⏸️ Queued"
        
        print(f"{priority_emoji.get(rec['priority'], '⚪')} {i}. {rec['title']}")
        print(f"   Type: {rec['material_type'].replace('_', ' ').title()}")
        print(f"   Priority: {rec['priority'].upper()} | Time: {rec['estimated_time_minutes']}min")
        print(f"   Status: {selected}")
        print(f"   Description: {rec['description']}")
        print()
    
    print(f"⏱️ TIMING SUMMARY:")
    print(f"   • Total available time: 120 minutes (2 hours)")
    print(f"   • Time for selected materials: {total_time} minutes")
    print(f"   • Remaining time: {120 - total_time} minutes")
    
    print(f"\n🚀 GENERATION QUEUE:")
    for i, task in enumerate(generation_queue, 1):
        print(f"{i}. {task['title']} ({task['type']})")
        print(f"   Task ID: {task['task_id'][:8]}...")
        print(f"   Priority: {task['priority']} | Status: {task['status']}")
        print(f"   Instructions: {task['instructions'][:80]}...")
        print()
    
    print("✅ MATERIALS DECISION WORKFLOW COMPLETE!")
    print("\nNext steps:")
    print("   1. Send generation queue to image/video agents")
    print("   2. Track creation progress")
    print("   3. Review and approve generated materials")
    print("   4. Integrate into final presentation")
    
    return {
        "recommendations": recommendations,
        "selected_materials": selected_materials,
        "generation_queue": generation_queue,
        "total_time": total_time
    }

if __name__ == "__main__":
    result = demo_simple_materials_agent()
    
    print(f"\n" + "=" * 70)
    print("📈 DEMO SUMMARY:")
    print(f"   • Recommendations generated: {len(result['recommendations'])}")
    print(f"   • Materials selected: {len(result['selected_materials'])}")
    print(f"   • Tasks queued: {len(result['generation_queue'])}")
    print(f"   • Estimated creation time: {result['total_time']} minutes")
    print("=" * 70)