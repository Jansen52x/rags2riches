"""
Enhanced Video Generator with Animations
Creates visually appealing animated videos from data, not just static images
"""
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from moviepy import VideoClip, CompositeVideoClip, concatenate_videoclips, TextClip
from moviepy import vfx
from PIL import Image, ImageDraw, ImageFont
import tempfile
import json


class AnimatedVideoGenerator:
    """Generate animated videos with dynamic charts and effects"""
    
    def __init__(self, output_dir: str = "generated_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = 1920  # Full HD
        self.height = 1080
        self.fps = 30
        
    def create_presentation_video(self, spec: Dict[str, Any]) -> str:
        """
        Create an animated presentation video from visualization specs
        
        Args:
            spec: Dictionary containing:
                - title: Presentation title
                - client_name: Client name
                - sections: List of visualization specs to animate
                - duration_per_section: Duration for each section (default: 5s)
        
        Returns:
            Path to generated MP4 file
        """
        sections = spec.get('sections', [])
        title = spec.get('title', 'Presentation')
        client_name = spec.get('client_name', 'client')
        client_slug = client_name.replace(' ', '_')
        duration_per_section = spec.get('duration_per_section', 5)
        
        clips = []
        
        # Title slide with fade in
        print("   Creating title slide...")
        title_clip = self._create_animated_title(title, duration=3)
        clips.append(title_clip)
        
        # Create animated clips for each section
        for i, section in enumerate(sections, 1):
            print(f"   Animating section {i}/{len(sections)}: {section.get('type', 'unknown')}...")
            
            section_clip = self._create_animated_section(
                section, 
                duration=duration_per_section
            )
            
            if section_clip:
                # Add clips (fade effects can be added later if needed)
                clips.append(section_clip)
        
        if not clips:
            raise ValueError("No clips were generated")
        
        # Concatenate all clips
        print("   Combining all sections...")
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"{client_slug}_animated_presentation_{timestamp}.mp4"
        
        # Write video file
        print("   Rendering final video (this may take a minute)...")
        final_video.write_videofile(
            str(filename),
            fps=self.fps,
            codec='libx264',
            audio=False,
            preset='medium',
            logger=None
        )
        
        # Cleanup
        final_video.close()
        for clip in clips:
            clip.close()
        
        return str(filename)
    
    def _create_animated_title(self, title: str, duration: float = 3) -> VideoClip:
        """Create an animated title slide with fade and zoom effect"""
        
        def make_frame(t):
            # Create frame
            img = Image.new('RGB', (self.width, self.height), color='#1a1a2e')
            draw = ImageDraw.Draw(img)
            
            # Load font
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
                subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
            except:
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
            
            # Animate text with fade and slight zoom
            progress = min(t / 1.0, 1.0)  # Fade in over 1 second
            alpha = int(255 * progress)
            scale = 0.8 + (0.2 * progress)  # Zoom from 80% to 100%
            
            # Draw title
            bbox = draw.textbbox((0, 0), title, font=title_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (self.width - text_width) // 2
            y = (self.height - text_height) // 2 - 50
            
            # Create overlay for fade effect
            overlay = Image.new('RGBA', (self.width, self.height), (26, 26, 46, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.text((x, y), title, fill=(255, 255, 255, alpha), font=title_font)
            
            # Composite
            img.paste(overlay, (0, 0), overlay)
            
            # Add accent line
            if progress > 0.5:
                line_width = int((self.width * 0.3) * ((progress - 0.5) * 2))
                line_y = y + text_height + 30
                line_x_start = (self.width - line_width) // 2
                draw.rectangle(
                    [(line_x_start, line_y), (line_x_start + line_width, line_y + 4)],
                    fill='#00adb5'
                )
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def _create_animated_section(self, section: Dict, duration: float) -> VideoClip:
        """Create an animated visualization based on section type"""
        
        section_type = section.get('type')
        
        if section_type == 'market_share':
            return self._animate_market_share(section, duration)
        elif section_type == 'swot_analysis':
            return self._animate_swot(section, duration)
        elif section_type == 'competitive_matrix':
            return self._animate_competitive_matrix(section, duration)
        elif section_type == 'growth_trend':
            return self._animate_growth_trend(section, duration)
        else:
            # Fallback: create static slide
            return self._create_static_slide(section, duration)
    
    def _animate_market_share(self, section: Dict, duration: float) -> VideoClip:
        """Animate a market share bar chart with bars growing"""
        
        data = section['data']
        companies = data['companies']
        market_shares = data['market_share']
        title = section.get('title', 'Market Share Analysis')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        def make_frame(t):
            # Animation progress
            progress = min(t / (duration * 0.6), 1.0)  # Animate in first 60% of duration
            progress = self._ease_out_cubic(progress)  # Smooth easing
            
            # Create frame
            img = Image.new('RGB', (self.width, self.height), color='#ffffff')
            draw = ImageDraw.Draw(img)
            
            # Draw title
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            
            draw.text((100, 80), title, fill='#1a1a2e', font=title_font)
            
            # Chart area
            chart_left = 300
            chart_top = 300
            chart_width = self.width - 600
            chart_height = self.height - 500
            max_value = max(market_shares)
            
            bar_width = chart_width // len(companies) - 40
            
            for i, (company, share) in enumerate(zip(companies, market_shares)):
                # Animated bar height
                animated_share = share * progress
                bar_height = (animated_share / max_value) * chart_height
                
                x = chart_left + i * (chart_width // len(companies)) + 20
                y = chart_top + chart_height - bar_height
                
                # Draw bar with color
                color = colors[i % len(colors)]
                draw.rectangle(
                    [(x, y), (x + bar_width, chart_top + chart_height)],
                    fill=color
                )
                
                # Draw value on top (appears when bar is mostly grown)
                if progress > 0.7:
                    value_alpha = int(255 * ((progress - 0.7) / 0.3))
                    value_text = f"{int(animated_share)}%"
                    bbox = draw.textbbox((0, 0), value_text, font=label_font)
                    text_width = bbox[2] - bbox[0]
                    
                    # Create overlay for text with alpha
                    overlay = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.text(
                        (x + (bar_width - text_width) // 2, y - 40),
                        value_text,
                        fill=(26, 26, 46, value_alpha),
                        font=label_font
                    )
                    img.paste(overlay, (0, 0), overlay)
                
                # Draw label
                draw.text((x, chart_top + chart_height + 20), company, fill='#1a1a2e', font=label_font)
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def _animate_swot(self, section: Dict, duration: float) -> VideoClip:
        """Animate SWOT analysis with items appearing one by one"""
        
        data = section['data']
        title = section.get('title', 'SWOT Analysis')
        
        strengths = data.get('strengths', [])[:4]
        weaknesses = data.get('weaknesses', [])[:4]
        opportunities = data.get('opportunities', [])[:4]
        threats = data.get('threats', [])[:4]
        
        quadrants = [
            ('Strengths', strengths, '#28a745'),
            ('Weaknesses', weaknesses, '#dc3545'),
            ('Opportunities', opportunities, '#007bff'),
            ('Threats', threats, '#ffc107')
        ]
        
        def make_frame(t):
            progress = t / duration
            
            img = Image.new('RGB', (self.width, self.height), color='#f5f5f5')
            draw = ImageDraw.Draw(img)
            
            # Title
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
                item_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            except:
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                item_font = ImageFont.load_default()
            
            draw.text((self.width // 2 - 200, 60), title, fill='#1a1a2e', font=title_font)
            
            # Quadrants
            quad_width = (self.width - 300) // 2
            quad_height = (self.height - 350) // 2
            positions = [
                (100, 200),  # Top-left
                (self.width // 2 + 50, 200),  # Top-right
                (100, self.height // 2 + 75),  # Bottom-left
                (self.width // 2 + 50, self.height // 2 + 75)  # Bottom-right
            ]
            
            items_per_quad = 4
            total_items = len(quadrants) * items_per_quad
            items_to_show = int(progress * total_items * 1.2)  # Slightly faster reveal
            
            item_count = 0
            
            for idx, ((quad_title, items, color), (x, y)) in enumerate(zip(quadrants, positions)):
                # Draw quadrant background
                draw.rectangle(
                    [(x, y), (x + quad_width, y + quad_height)],
                    fill='white',
                    outline=color,
                    width=3
                )
                
                # Draw header
                draw.text((x + 20, y + 20), quad_title, fill=color, font=header_font)
                
                # Draw items with animation
                for i, item in enumerate(items):
                    if item_count < items_to_show:
                        # Fade in effect for each item
                        item_progress = min((items_to_show - item_count), 1.0)
                        alpha = int(255 * item_progress)
                        
                        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        
                        item_y = y + 80 + i * 60
                        overlay_draw.text(
                            (x + 30, item_y),
                            f"• {item}",
                            fill=(26, 26, 46, alpha),
                            font=item_font
                        )
                        
                        img.paste(overlay, (0, 0), overlay)
                    
                    item_count += 1
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def _animate_growth_trend(self, section: Dict, duration: float) -> VideoClip:
        """Animate line chart with lines drawing progressively"""
        
        data = section['data']
        title = section.get('title', 'Growth Trends')
        years = data['years']
        entities = data['entities']
        
        def make_frame(t):
            progress = min(t / (duration * 0.7), 1.0)
            progress = self._ease_out_cubic(progress)
            
            img = Image.new('RGB', (self.width, self.height), color='#ffffff')
            draw = ImageDraw.Draw(img)
            
            # Title
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            
            draw.text((100, 80), title, fill='#1a1a2e', font=title_font)
            
            # Chart area
            chart_left = 200
            chart_top = 250
            chart_width = self.width - 400
            chart_height = self.height - 450
            
            # Find max value for scaling
            all_values = [val for entity in entities for val in entity['values']]
            max_val = max(all_values)
            min_val = min(all_values)
            
            # Draw axes
            draw.line([(chart_left, chart_top + chart_height), 
                      (chart_left + chart_width, chart_top + chart_height)], 
                     fill='#666', width=2)
            draw.line([(chart_left, chart_top), 
                      (chart_left, chart_top + chart_height)], 
                     fill='#666', width=2)
            
            # Draw year labels
            for i, year in enumerate(years):
                x = chart_left + (i * chart_width / (len(years) - 1))
                draw.text((x - 20, chart_top + chart_height + 20), str(year), 
                         fill='#1a1a2e', font=label_font)
            
            # Draw lines for each entity
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            points_to_draw = int(progress * len(years))
            
            for entity_idx, entity in enumerate(entities):
                color = colors[entity_idx % len(colors)]
                name = entity['name']
                values = entity['values']
                
                # Draw line progressively
                for i in range(min(points_to_draw, len(values) - 1)):
                    x1 = chart_left + (i * chart_width / (len(years) - 1))
                    y1 = chart_top + chart_height - ((values[i] - min_val) / (max_val - min_val) * chart_height)
                    x2 = chart_left + ((i + 1) * chart_width / (len(years) - 1))
                    y2 = chart_top + chart_height - ((values[i + 1] - min_val) / (max_val - min_val) * chart_height)
                    
                    draw.line([(x1, y1), (x2, y2)], fill=color, width=4)
                    draw.ellipse([(x1 - 6, y1 - 6), (x1 + 6, y1 + 6)], fill=color)
                
                # Draw last point
                if points_to_draw >= len(values):
                    x = chart_left + ((len(values) - 1) * chart_width / (len(years) - 1))
                    y = chart_top + chart_height - ((values[-1] - min_val) / (max_val - min_val) * chart_height)
                    draw.ellipse([(x - 6, y - 6), (x + 6, y + 6)], fill=color)
                
                # Draw legend
                legend_x = chart_left + chart_width - 300
                legend_y = chart_top + 50 + entity_idx * 50
                draw.rectangle([(legend_x, legend_y), (legend_x + 40, legend_y + 20)], fill=color)
                draw.text((legend_x + 50, legend_y), name, fill='#1a1a2e', font=label_font)
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def _animate_competitive_matrix(self, section: Dict, duration: float) -> VideoClip:
        """Animate competitive positioning with dots appearing and labels"""
        
        data = section['data']
        title = section.get('title', 'Competitive Positioning')
        competitors = data['competitors']
        
        def make_frame(t):
            progress = t / duration
            
            img = Image.new('RGB', (self.width, self.height), color='#ffffff')
            draw = ImageDraw.Draw(img)
            
            # Title
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
                label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
                axis_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
                axis_font = ImageFont.load_default()
            
            draw.text((100, 60), title, fill='#1a1a2e', font=title_font)
            
            # Chart area
            chart_size = min(self.width, self.height) - 400
            chart_left = (self.width - chart_size) // 2
            chart_top = 250
            
            # Draw axes
            mid_x = chart_left + chart_size // 2
            mid_y = chart_top + chart_size // 2
            
            draw.line([(chart_left, mid_y), (chart_left + chart_size, mid_y)], 
                     fill='#999', width=2)
            draw.line([(mid_x, chart_top), (mid_x, chart_top + chart_size)], 
                     fill='#999', width=2)
            
            # Axis labels
            x_label = data.get('x_axis_label', 'Capability')
            y_label = data.get('y_axis_label', 'Value')
            
            draw.text((chart_left + chart_size // 2 - 100, chart_top + chart_size + 30), 
                     x_label, fill='#1a1a2e', font=axis_font)
            
            # Animate competitors appearing
            for i, comp in enumerate(competitors):
                comp_progress = max(0, min(1, (progress - i * 0.15) / 0.3))
                
                if comp_progress > 0:
                    # Scale position to chart
                    x = chart_left + (comp['x'] / 10) * chart_size
                    y = chart_top + chart_size - (comp['y'] / 10) * chart_size
                    
                    # Animated size
                    size = int(40 * comp_progress)
                    color = comp.get('color', '#2E86AB')
                    
                    # Draw dot
                    draw.ellipse([(x - size, y - size), (x + size, y + size)], 
                               fill=color, outline='white', width=3)
                    
                    # Draw label (appears after dot)
                    if comp_progress > 0.7:
                        label_alpha = int(255 * ((comp_progress - 0.7) / 0.3))
                        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        overlay_draw.text((x + size + 10, y - 15), comp['name'], 
                                        fill=(26, 26, 46, label_alpha), font=label_font)
                        img.paste(overlay, (0, 0), overlay)
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    def _create_static_slide(self, section: Dict, duration: float) -> VideoClip:
        """Fallback: create a simple static slide"""
        
        title = section.get('title', 'Information')
        
        def make_frame(t):
            img = Image.new('RGB', (self.width, self.height), color='#f5f5f5')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
            except:
                font = ImageFont.load_default()
            
            draw.text((100, self.height // 2), title, fill='#1a1a2e', font=font)
            
            return np.array(img)
        
        return VideoClip(make_frame, duration=duration)
    
    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """Cubic easing function for smooth animations"""
        return 1 - pow(1 - t, 3)
