import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json
from moviepy import ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

class ContentGenerator:
    """Generates charts, images, and visualizations for sales meetings"""
    
    def __init__(self, output_dir: str = "generated_content"):
        # Use absolute path relative to this file's location
        # This ensures content is saved in image-video-agent/generated_content
        # regardless of where the script is run from
        if not Path(output_dir).is_absolute():
            generator_dir = Path(__file__).parent
            self.output_dir = generator_dir / output_dir
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate(self, spec: Dict[str, Any]) -> str:
        """
        Main generation method. Routes to appropriate generator based on spec type.
        
        Args:
            spec: Dictionary with 'type' and relevant data
            
        Returns:
            Path to generated file
        """
        content_type = spec.get('type')
        
        generators = {
            'market_share': self._generate_market_share,
            'growth_trend': self._generate_growth_trend,
            'competitive_matrix': self._generate_competitive_matrix,
            'swot_analysis': self._generate_swot_analysis,
            'financial_comparison': self._generate_financial_comparison,
            'industry_trends': self._generate_industry_trends,
            'video_presentation': self._generate_video_presentation,
            'animated_video': self._generate_animated_video,
        }
        
        if content_type not in generators:
            raise ValueError(f"Unknown content type: {content_type}")
        
        return generators[content_type](spec)
    
    def _generate_market_share(self, spec: Dict) -> str:
        """Generate market share bar chart"""
        data = spec['data']
        
        fig = go.Figure(data=[
            go.Bar(
                x=data['companies'],
                y=data['market_share'],
                marker_color=data.get('colors', ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']),
                text=data['market_share'],
                texttemplate='%{text}%',
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title={
                'text': spec.get('title', 'Market Share Analysis'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            yaxis_title='Market Share (%)',
            template='plotly_white',
            height=600,
            font=dict(size=14),
            showlegend=False
        )
        
        filename = self._get_filename('market_share', spec)
        fig.write_image(filename, width=1200, height=600)
        return str(filename)
    
    def _generate_growth_trend(self, spec: Dict) -> str:
        """Generate line chart showing growth trends"""
        data = spec['data']
        
        fig = go.Figure()
        
        # Add trace for each entity (client, competitors, industry avg)
        for entity in data['entities']:
            fig.add_trace(go.Scatter(
                x=data['years'],
                y=entity['values'],
                mode='lines+markers',
                name=entity['name'],
                line=dict(width=3),
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title={
                'text': spec.get('title', 'Growth Trend Analysis'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title='Year',
            yaxis_title=data.get('y_axis_label', 'Revenue ($M)'),
            template='plotly_white',
            hovermode='x unified',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        filename = self._get_filename('growth_trend', spec)
        fig.write_image(filename, width=1200, height=600)
        return str(filename)
    
    def _generate_competitive_matrix(self, spec: Dict) -> str:
        """Generate 2x2 competitive positioning matrix"""
        data = spec['data']
        
        fig = go.Figure()
        
        for competitor in data['competitors']:
            fig.add_trace(go.Scatter(
                x=[competitor['x']],
                y=[competitor['y']],
                mode='markers+text',
                name=competitor['name'],
                text=competitor['name'],
                textposition='top center',
                marker=dict(
                    size=25,
                    color=competitor.get('color', '#2E86AB'),
                    line=dict(width=2, color='white')
                ),
                textfont=dict(size=12, color='black')
            ))
        
        # Add quadrant lines
        x_mid = data.get('x_midpoint', 5)
        y_mid = data.get('y_midpoint', 5)
        
        fig.add_hline(y=y_mid, line_dash="dash", line_color="gray", opacity=0.5, line_width=2)
        fig.add_vline(x=x_mid, line_dash="dash", line_color="gray", opacity=0.5, line_width=2)
        
        # Add quadrant labels
        fig.add_annotation(x=2.5, y=7.5, text="High Value<br>Low Capability", 
                          showarrow=False, font=dict(size=11, color='gray'))
        fig.add_annotation(x=7.5, y=7.5, text="High Value<br>High Capability", 
                          showarrow=False, font=dict(size=11, color='gray'))
        fig.add_annotation(x=2.5, y=2.5, text="Low Value<br>Low Capability", 
                          showarrow=False, font=dict(size=11, color='gray'))
        fig.add_annotation(x=7.5, y=2.5, text="Low Value<br>High Capability", 
                          showarrow=False, font=dict(size=11, color='gray'))
        
        fig.update_layout(
            title={
                'text': spec.get('title', 'Competitive Positioning Matrix'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title=data.get('x_axis_label', 'Capability'),
            yaxis_title=data.get('y_axis_label', 'Strategic Value'),
            xaxis=dict(range=[0, 10], showgrid=True),
            yaxis=dict(range=[0, 10], showgrid=True),
            template='plotly_white',
            showlegend=False,
            height=700
        )
        
        filename = self._get_filename('competitive_matrix', spec)
        fig.write_image(filename, width=1000, height=800)
        return str(filename)
    
    def _generate_swot_analysis(self, spec: Dict) -> str:
        """Generate SWOT analysis visualization"""
        data = spec['data']
        
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Strengths', 'Weaknesses', 'Opportunities', 'Threats'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        quadrants = [
            (data.get('strengths', []), 1, 1, '#28a745'),
            (data.get('weaknesses', []), 1, 2, '#dc3545'),
            (data.get('opportunities', []), 2, 1, '#007bff'),
            (data.get('threats', []), 2, 2, '#ffc107')
        ]
        
        for items, row, col, color in quadrants:
            # Create text annotation for each item
            text_content = '<br>'.join([f"• {item}" for item in items[:5]])  # Top 5 items
            
            fig.add_annotation(
                text=text_content,
                xref=f"x{row+col-1 if row+col > 2 else ''}", 
                yref=f"y{row+col-1 if row+col > 2 else ''}",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12),
                align='left',
                row=row, col=col
            )
        
        fig.update_layout(
            title={
                'text': spec.get('title', f"SWOT Analysis: {spec.get('company_name', 'Client')}"),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=800,
            showlegend=False
        )
        
        # Hide axes
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        filename = self._get_filename('swot_analysis', spec)
        fig.write_image(filename, width=1200, height=800)
        return str(filename)
    
    def _generate_financial_comparison(self, spec: Dict) -> str:
        """Generate grouped bar chart for financial metrics comparison"""
        data = spec['data']
        
        fig = go.Figure()
        
        for entity in data['entities']:
            fig.add_trace(go.Bar(
                name=entity['name'],
                x=data['metrics'],
                y=entity['values'],
                text=entity['values'],
                texttemplate='$%{text}M',
                textposition='outside'
            ))
        
        fig.update_layout(
            title={
                'text': spec.get('title', 'Financial Metrics Comparison'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title='Metrics',
            yaxis_title='Value ($M)',
            barmode='group',
            template='plotly_white',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        filename = self._get_filename('financial_comparison', spec)
        fig.write_image(filename, width=1200, height=600)
        return str(filename)
    
    def _generate_industry_trends(self, spec: Dict) -> str:
        """Generate multi-line chart for industry trends"""
        data = spec['data']
        
        fig = go.Figure()
        
        for trend in data['trends']:
            fig.add_trace(go.Scatter(
                x=data['periods'],
                y=trend['values'],
                mode='lines+markers',
                name=trend['name'],
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title={
                'text': spec.get('title', 'Industry Trends'),
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            xaxis_title=data.get('x_axis_label', 'Period'),
            yaxis_title=data.get('y_axis_label', 'Value'),
            template='plotly_white',
            hovermode='x unified',
            height=600
        )
        
        filename = self._get_filename('industry_trends', spec)
        fig.write_image(filename, width=1200, height=600)
        return str(filename)
    
    def _get_filename(self, content_type: str, spec: Dict) -> Path:
        """Generate unique filename for output"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        client_name = spec.get('client_name', 'client').replace(' ', '_')
        return self.output_dir / f"{client_name}_{content_type}_{timestamp}.png"    
    def _generate_video_presentation(self, spec: Dict) -> str:
        """
        Generate a video presentation from a list of images
        
        Args:
            spec: Dictionary containing:
                - image_files: List of image file paths to include
                - title: Optional title for the video
                - duration_per_slide: Duration in seconds for each slide (default: 3)
                - fps: Frames per second (default: 24)
        
        Returns:
            Path to generated MP4 file
        """
        data = spec['data']
        image_files = data.get('image_files', [])
        
        if not image_files:
            raise ValueError("No image files provided for video generation")
        
        duration_per_slide = data.get('duration_per_slide', 3)
        fps = data.get('fps', 24)
        
        clips = []
        
        # Add title slide if provided
        if spec.get('title'):
            title_clip = self._create_title_slide(
                spec['title'],
                duration=2,
                fps=fps
            )
            clips.append(title_clip)
        
        # Add image slides
        for img_path in image_files:
            if not Path(img_path).exists():
                print(f"⚠️  Warning: Image not found: {img_path}")
                continue
            
            clip = ImageClip(str(img_path), duration=duration_per_slide)
            clips.append(clip)
        
        if not clips:
            raise ValueError("No valid images found to create video")
        
        # Concatenate all clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        client_name = spec.get('client_name', 'client').replace(' ', '_')
        filename = self.output_dir / f"{client_name}_video_presentation_{timestamp}.mp4"
        
        # Write video file
        final_video.write_videofile(
            str(filename),
            fps=fps,
            codec='libx264',
            audio=False,
            logger=None
        )
        
        # Close clips to free memory
        final_video.close()
        for clip in clips:
            clip.close()
        
        return str(filename)
    
    def _create_title_slide(self, title: str, duration: float = 2, fps: int = 24) -> ImageClip:
        """
        Create a title slide as an image
        
        Args:
            title: Title text to display
            duration: Duration of the slide in seconds
            fps: Frames per second
        
        Returns:
            ImageClip with the title
        """
        # Create a blank image
        width, height = 1200, 600
        img = Image.new('RGB', (width, height), color='#2E86AB')
        draw = ImageDraw.Draw(img)
        
        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 60)
            except:
                font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), title, fill='white', font=font)
        
        # Save temporary image
        temp_path = self.output_dir / "temp_title_slide.png"
        img.save(temp_path)
        
        # Create clip from image
        clip = ImageClip(str(temp_path), duration=duration)
        
        return clip
    
    def _generate_animated_video(self, spec: Dict) -> str:
        """
        Generate an animated video presentation with dynamic visualizations
        
        Args:
            spec: Dictionary containing:
                - sections: List of visualization specs to animate
                - title: Video title
                - client_name: Client name
                - duration_per_section: Duration for each section
        
        Returns:
            Path to generated MP4 file
        """
        from animated_video_generator import AnimatedVideoGenerator
        
        video_gen = AnimatedVideoGenerator(output_dir=str(self.output_dir))
        return video_gen.create_presentation_video(spec)
