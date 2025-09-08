from flask import Flask, render_template, Response, request, jsonify, send_file, session, redirect, url_for
import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime
from io import BytesIO
import time
import threading
from queue import Queue
import os
from functools import wraps
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define application root directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'authenticated' not in session or not session['authenticated']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
app.secret_key = 'attendance-system-secret-key-2024'

# File paths
ENCODINGS_FILE = os.path.join(APP_ROOT, "encodings.pkl")
USERS_FILE = os.path.join(APP_ROOT, "users.json")
DETAILS_FILE = os.path.join(APP_ROOT, "details.json")

# Section configurations
SECTIONS = {
    "CSE_DS": {"prefix": "23CSEDS", "start": 1, "end": 60, "name": "CSE-DS"},
    "CSEAIML_A": {"prefix": "23CSEAIML", "start": 0, "end": 64, "name": "CSE AIML-A"},
    "CSEAIML_B": {"prefix": "23CSEAIML", "start": 65, "end": 128, "name": "CSE AIML-B"},
    "CSEAIML_C": {"prefix": "23CSEAIML", "start": 129, "end": 192, "name": "CSE AIML-C"}
}

# Timetable configuration - Days and subjects for each section
TIMETABLE = {
    "CSE_DS": {
        "Monday": ["Data Structures", "Python Programming", "Database Systems", "Statistics"],
        "Tuesday": ["Machine Learning", "Data Visualization", "Python Programming", "Mathematics"],
        "Wednesday": ["Database Systems", "Statistics", "Data Structures", "Communication Skills"],
        "Thursday": ["Data Visualization", "Machine Learning", "Mathematics", "Python Programming"],
        "Friday": ["Statistics", "Data Structures", "Database Systems", "Machine Learning"],
        "Saturday": ["Mathematics", "Communication Skills", "Data Visualization", "Python Programming"]
    },
    "CSEAIML_A": {
        "Monday": ["Artificial Intelligence", "Deep Learning", "Python Programming", "Mathematics"],
        "Tuesday": ["Machine Learning", "Neural Networks", "Data Structures", "Communication Skills"],
        "Wednesday": ["Deep Learning", "Mathematics", "Artificial Intelligence", "Python Programming"],
        "Thursday": ["Neural Networks", "Machine Learning", "Data Structures", "Deep Learning"],
        "Friday": ["Mathematics", "Artificial Intelligence", "Machine Learning", "Neural Networks"],
        "Saturday": ["Python Programming", "Communication Skills", "Data Structures", "Artificial Intelligence"]
    },
    "CSEAIML_B": {
        "Monday": ["Machine Learning", "Neural Networks", "Python Programming", "Mathematics"],
        "Tuesday": ["Artificial Intelligence", "Deep Learning", "Data Structures", "Communication Skills"],
        "Wednesday": ["Neural Networks", "Mathematics", "Machine Learning", "Python Programming"],
        "Thursday": ["Deep Learning", "Artificial Intelligence", "Data Structures", "Neural Networks"],
        "Friday": ["Mathematics", "Machine Learning", "Artificial Intelligence", "Deep Learning"],
        "Saturday": ["Python Programming", "Communication Skills", "Data Structures", "Machine Learning"]
    },
    "CSEAIML_C": {
        "Monday": ["Deep Learning", "Artificial Intelligence", "Python Programming", "Mathematics"],
        "Tuesday": ["Neural Networks", "Machine Learning", "Data Structures", "Communication Skills"],
        "Wednesday": ["Artificial Intelligence", "Mathematics", "Deep Learning", "Python Programming"],
        "Thursday": ["Machine Learning", "Neural Networks", "Data Structures", "Artificial Intelligence"],
        "Friday": ["Mathematics", "Deep Learning", "Neural Networks", "Machine Learning"],
        "Saturday": ["Python Programming", "Communication Skills", "Data Structures", "Deep Learning"]
    }
}

# Global variables for camera processing
camera_processor = None
present_students = set()
attendance_started = False

# Import for SGPA graph
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. SGPA graphs will be disabled.")

def create_enhanced_sgpa_graph(sgpas):
    """Create an enhanced SGPA graph using Plotly"""
    if not PLOTLY_AVAILABLE:
        return None
        
    # Extract semester numbers and SGPA values
    semesters = []
    sgpa_values = []
    colors = []
    
    for semester, sgpa in sgpas.items():
        if sgpa:  # Skip empty values
            semesters.append(f"Semester {semester}")
            sgpa_value = float(sgpa)
            sgpa_values.append(sgpa_value)
            
            # Color based on SGPA value
            if sgpa_value >= 8.5:
                colors.append('#28a745')  # Green for excellent
            elif sgpa_value >= 7.5:
                colors.append('#17a2b8')  # Blue for good
            elif sgpa_value >= 6.5:
                colors.append('#ffc107')  # Yellow for average
            else:
                colors.append('#dc3545')  # Red for below average
    
    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=semesters,
            y=sgpa_values,
            marker=dict(
                color=colors,
                line=dict(width=1.5, color='rgba(0,0,0,0.3)')
            ),
            hovertemplate='<b>%{x}</b><br>SGPA: %{y:.2f}<extra></extra>',
            name='SGPA'
        )
    )
    
    # Add a line for the trend
    fig.add_trace(
        go.Scatter(
            x=semesters,
            y=sgpa_values,
            mode='lines+markers',
            line=dict(color='rgba(0,0,0,0.7)', width=2, dash='dot'),
            marker=dict(size=8, symbol='circle', line=dict(width=2, color='rgba(0,0,0,0.7)')),
            name='Trend',
            hovertemplate='<b>%{x}</b><br>SGPA: %{y:.2f}<extra></extra>'
        )
    )
    
    # Calculate CGPA
    cgpa = sum(sgpa_values) / len(sgpa_values) if sgpa_values else 0
    
    # Add a horizontal line for CGPA
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=cgpa,
        x1=len(semesters) - 0.5,
        y1=cgpa,
        line=dict(color="rgba(255,0,0,0.7)", width=2, dash="dash")
    )
    
    # Add annotation for CGPA
    fig.add_annotation(
        x=len(semesters) - 1,
        y=cgpa,
        text=f"CGPA: {cgpa:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(255,0,0,0.7)",
        ax=40,
        ay=-40
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Semester-wise SGPA Performance',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Semester",
        yaxis_title="SGPA",
        yaxis=dict(
            range=[0, 10],  # SGPA is typically on a scale of 0-10
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    # Return the HTML representation of the figure
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def get_student_image_url(roll_no):
    """Get student image URL"""
    return f"https://gietuerp.in/StudentDocuments/{roll_no}/{roll_no}.JPG"

def get_student_attendance_history(roll_number):
    """Get attendance history for a specific student"""
    attendance_data = load_attendance_data()
    history = []
    
    # Find which section this student belongs to
    student_section = None
    for section_id, section_config in SECTIONS.items():
        prefix = section_config["prefix"]
        try:
            roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
            if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                student_section = section_id
                break
        except (ValueError, TypeError):
            continue
    
    if not student_section:
        return history  # Empty history if section not found
    
    # Get attendance records for this student in their section
    if student_section in attendance_data:
        section_data = attendance_data[student_section]
        for date, students in section_data.items():
            if roll_number in students:
                status = "Present" if students[roll_number] == 1 else "Absent"
                history.append({
                    "date": date,
                    "status": status,
                    "section": SECTIONS[student_section]["name"]
                })
    
    # Sort by date (newest first)
    history.sort(key=lambda x: x["date"], reverse=True)
    return history

def get_student_daily_attendance(roll_number):
    """Get daily class attendance for a specific student"""
    attendance_data = load_attendance_data()
    daily_attendance = []
    
    # Find which section this student belongs to
    student_section = None
    for section_id, section_config in SECTIONS.items():
        prefix = section_config["prefix"]
        try:
            roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
            if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                student_section = section_id
                break
        except (ValueError, TypeError):
            continue
    
    if not student_section:
        return daily_attendance  # Empty if section not found
    
    # Group attendance by date and calculate classes attended vs total classes
    if student_section in attendance_data:
        # Get the timetable for this section
        section_timetable = TIMETABLE.get(student_section, {})
        
        # Group by date
        dates = {}
        for date, students in attendance_data[student_section].items():
            day_of_week = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            total_classes = len(section_timetable.get(day_of_week, []))
            
            if total_classes == 0:
                continue  # Skip dates with no classes scheduled
                
            # Initialize the date entry if not exists
            if date not in dates:
                dates[date] = {
                    "date": date,
                    "day": day_of_week,
                    "classes_attended": 0,
                    "total_classes": total_classes
                }
            
            # Count this student's attendance for this date
            if roll_number in students and students[roll_number] == 1:
                dates[date]["classes_attended"] += 1
        
        # Convert to list and sort by date (newest first)
        daily_attendance = list(dates.values())
        daily_attendance.sort(key=lambda x: x["date"], reverse=True)
    
    return daily_attendance

def calculate_attendance_percentage(roll_number):
    """Calculate overall attendance percentage for a student"""
    daily_attendance = get_student_daily_attendance(roll_number)
    
    if not daily_attendance:
        return 0  # No attendance records
    
    total_classes = sum(day["total_classes"] for day in daily_attendance)
    classes_attended = sum(day["classes_attended"] for day in daily_attendance)
    
    if total_classes == 0:
        return 0  # Avoid division by zero
        
    return round((classes_attended / total_classes) * 100, 1)
    
    return history

def get_student_daily_attendance(roll_number):
    """Get daily attendance data for a specific student"""
    daily_attendance = load_daily_attendance()
    history = []
    
    # Find which section this student belongs to
    student_section = None
    for section_id, section_config in SECTIONS.items():
        prefix = section_config["prefix"]
        try:
            roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
            if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                student_section = section_id
                break
        except (ValueError, TypeError):
            continue
    
    if not student_section:
        return history  # Empty history if section not found
    
    # If no real data or empty data, generate dummy data for 6 days with the timetable
    if student_section not in daily_attendance or not daily_attendance[student_section]:
        if student_section in TIMETABLE:
            import random
            from datetime import datetime, timedelta
            
            today = datetime.now()
            for i in range(6):
                day = today - timedelta(days=i)
                day_name = day.strftime('%A')
                if day_name in TIMETABLE[student_section]:
                    total_classes = len(TIMETABLE[student_section][day_name])
                    # Random attendance between 0 and total classes
                    attended = random.randint(0, total_classes)
                    history.append({
                        "date": day.strftime('%Y-%m-%d'),
                        "classes_attended": attended,
                        "total_classes": total_classes,
                        "ratio": f"{attended}/{total_classes}",
                        "section": SECTIONS[student_section]["name"],
                        "subjects": TIMETABLE[student_section][day_name]
                    })
            return sorted(history, key=lambda x: x["date"], reverse=True)
        return history  # Empty history if no timetable
    
    # Get daily attendance records for this student in their section
    section_data = daily_attendance[student_section]
    for date, students in section_data.items():
        if roll_number in students:
            attended = students[roll_number]["attended"]
            total = students[roll_number]["total"]
            
            # Get day of week for this date
            day_name = datetime.strptime(date, '%Y-%m-%d').strftime('%A')
            subjects = TIMETABLE[student_section].get(day_name, [])
            
            history.append({
                "date": date,
                "classes_attended": attended,
                "total_classes": total,
                "ratio": f"{attended}/{total}",
                "section": SECTIONS[student_section]["name"],
                "subjects": subjects
            })
    
    # Sort by date (newest first)
    history.sort(key=lambda x: x["date"], reverse=True)
    
    return history

def calculate_attendance_percentage(roll_number):
    """Calculate attendance percentage based on daily attendance"""
    daily_attendance = load_daily_attendance()
    
    # Find which section this student belongs to
    student_section = None
    for section_id, section_config in SECTIONS.items():
        prefix = section_config["prefix"]
        try:
            roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
            if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                student_section = section_id
                break
        except (ValueError, TypeError):
            continue
    
    if not student_section or student_section not in daily_attendance:
        return 0  # Return 0% if section not found
    
    total_attended = 0
    total_classes = 0
    
    # Calculate total attended and total classes
    section_data = daily_attendance[student_section]
    for date, students in section_data.items():
        if roll_number in students:
            total_attended += students[roll_number]["attended"]
            total_classes += students[roll_number]["total"]
    
    # Calculate percentage
    if total_classes > 0:
        return round((total_attended / total_classes) * 100, 2)
    else:
        return 0

def get_sgpa_color(sgpa_val):
    """Get color based on SGPA value"""
    try:
        sgpa = float(sgpa_val)
        if sgpa >= 9.0:
            return "#27AE60"  # Green for excellent
        elif sgpa >= 8.0:
            return "#F39C12"  # Orange for good  
        elif sgpa >= 7.0:
            return "#3498DB"  # Blue for average
        else:
            return "#E74C3C"  # Red for below average
    except:
        return "#95A5A6"  # Gray for invalid

def create_enhanced_sgpa_graph(sgpas_dict, student_name):
    """Create an enhanced SGPA performance graph with fixed Plotly syntax"""
    if not PLOTLY_AVAILABLE or not sgpas_dict:
        return None
    
    # Prepare data
    semesters = []
    sgpa_values = []
    colors = []
    performance_labels = []
    
    for semester, sgpa in sorted(sgpas_dict.items(), key=lambda x: int(x[0])):
        try:
            sgpa_val = float(sgpa)
            semesters.append(f"Semester {semester}")
            sgpa_values.append(sgpa_val)
            colors.append(get_sgpa_color(sgpa))
            
            # Performance labels
            if sgpa_val >= 9.0:
                performance_labels.append("Excellent 🏆")
            elif sgpa_val >= 8.0:
                performance_labels.append("Good 🎯")
            elif sgpa_val >= 7.0:
                performance_labels.append("Average 📈")
            else:
                performance_labels.append("Needs Improvement 📚")
        except:
            continue
    
    if not semesters:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('SGPA Trend Analysis', 'Performance Distribution'),
        vertical_spacing=0.15,
        specs=[[{"secondary_y": False}],
               [{"type": "bar"}]]
    )
    
    # Main SGPA trend chart
    fig.add_trace(
        go.Scatter(
            x=semesters,
            y=sgpa_values,
            mode='lines+markers+text',
            line=dict(color='#4FACFE', width=4, shape='spline'),
            marker=dict(
                size=12,
                color=colors,
                line=dict(color='white', width=2),
                symbol='circle'
            ),
            text=[f"{val:.2f}" for val in sgpa_values],
            textposition="top center",
            textfont=dict(size=12, color='white', family="Arial Black"),
            hovertemplate='<b>%{x}</b><br>SGPA: %{y:.2f}<br>Performance: %{text}<extra></extra>',
            name='SGPA Trend',
            fill='tonexty',
            fillcolor='rgba(79, 172, 254, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add bar chart for performance distribution
    fig.add_trace(
        go.Bar(
            x=semesters,
            y=sgpa_values,
            marker=dict(color=colors),
            text=performance_labels,
            textposition="auto",
            name="Performance",
            hovertemplate='<b>%{x}</b><br>SGPA: %{y:.2f}<br>Performance: %{text}<extra></extra>',
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{student_name}'s Academic Performance",
        height=600,
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    
    # Update y-axis
    fig.update_yaxes(
        title_text="SGPA",
        range=[min(sgpa_values) - 0.5 if min(sgpa_values) > 0.5 else 0, 10],
        tickformat=".1f",
        gridcolor='rgba(0,0,0,0.1)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="SGPA",
        range=[0, 10],
        tickformat=".1f",
        gridcolor='rgba(0,0,0,0.1)',
        row=2, col=1
    )
    
    # Update x-axis
    fig.update_xaxes(
        title_text="",
        tickangle=0,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Semester",
        tickangle=0,
        row=2, col=1
    )
    
    return fig

# ----------------- Helper Functions -----------------
def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        default_users = {
            "dr.smith": {
                "password": "teacher123",
                "name": "Dr. John Smith"
            },
            "prof.johnson": {
                "password": "teacher456",
                "name": "Professor Sarah Johnson"
            },
            "admin": {
                "password": "admin@123",
                "name": "System Administrator"
            },
            "hod.cse": {
                "password": "hod2024",
                "name": "HOD Computer Science"
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f, indent=2)
        return default_users

# Student view route removed - functionality moved to modal in dashboard

def load_students_data():
    """Load students data with enhanced sample data for better graphs"""
    try:
        with open(DETAILS_FILE, 'r') as f:
            students_list = json.load(f)
            students_dict = {student['rollNo']: student for student in students_list}
            return students_dict
    except FileNotFoundError:
        # Enhanced sample data with more SGPA values for better graphs
        students_dict = {
            "22CSE998": {
                "rollNo": "22CSE998",
                "name": "ABUL HASAN",
                "mobile": "9835275387",
                "sgpas": {
                    "1": "8.5",
                    "2": "8.7",
                    "3": "7.35",
                    "4": "8.0",
                    "5": "8.2",
                    "6": "8.8"
                }
            },
            "23CSEAIML087": {
                "rollNo": "23CSEAIML087",
                "name": "RAHUL KUMAR",
                "mobile": "9876543210",
                "sgpas": {
                    "1": "9.0",
                    "2": "8.8",
                    "3": "9.2",
                    "4": "8.9",
                    "5": "9.1",
                    "6": "9.3"
                }
            },
            "23CSEDS015": {
                "rollNo": "23CSEDS015",
                "name": "PRIYA SHARMA",
                "mobile": "9123456789",
                "sgpas": {
                    "1": "7.8",
                    "2": "8.0",
                    "3": "7.9",
                    "4": "8.2",
                    "5": "8.4",
                    "6": "8.1"
                }
            }
        }
        return students_dict
    except Exception as e:
        logger.error(f"Error loading students data: {e}")
        return {}

def calculate_cgpa(sgpas_dict):
    """Calculate CGPA from SGPA values"""
    if not sgpas_dict:
        return 0.0
    
    total_sgpa = 0.0
    count = 0
    
    for sgpa_str in sgpas_dict.values():
        try:
            sgpa = float(sgpa_str)
            total_sgpa += sgpa
            count += 1
        except (ValueError, TypeError):
            continue
    
    return round(total_sgpa / count, 2) if count > 0 else 0.0

def load_encodings():
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    except FileNotFoundError:
        return [], []

def load_student_details():
    """Load student details from details.json"""
    try:
        with open(DETAILS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"Details file {DETAILS_FILE} not found. Using placeholder data.")
        # Create sample data for demonstration
        sample_data = []
        for section_key, section in SECTIONS.items():
            for i in range(section["start"], section["end"] + 1):
                roll_number = f"{section['prefix']}{i:03d}"
                sample_data.append({
                    "rollNo": roll_number,
                    "name": f"Student {roll_number}",
                    "mobile": f"9{np.random.randint(100000000, 999999999)}",
                    "sgpas": {
                        "1": round(np.random.uniform(7.0, 9.5), 2),
                        "2": round(np.random.uniform(7.5, 9.8), 2)
                    }
                })
        return sample_data
    except Exception as e:
        logger.error(f"Error loading student details: {e}")
        return []

def get_student_info(roll_number):
    """Get student details by roll number"""
    student_details = load_student_details()
    for student in student_details:
        if student['rollNo'] == roll_number:
            return student
    return {
        'rollNo': roll_number,
        'name': f"Student {roll_number}",
        'mobile': 'N/A',
        'sgpas': {}
    }

def get_section_students(section):
    config = SECTIONS[section]
    return [f"{config['prefix']}{i:03d}" for i in range(config["start"], config["end"]+1)]

def authenticate_user(username, password):
    users = load_users()
    user_data = users.get(username)
    if user_data and user_data.get('password') == password:
        # Make sure sections field exists
        if 'sections' not in user_data:
            user_data['sections'] = list(SECTIONS.keys())
        return user_data
    return None

def load_attendance_data():
    try:
        with open(os.path.join(APP_ROOT, 'attendance.json'), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Initialize with empty data for each section
        attendance_data = {section: {} for section in SECTIONS}
        with open(os.path.join(APP_ROOT, 'attendance.json'), 'w') as f:
            json.dump(attendance_data, f, indent=2)
        return attendance_data

def load_timetable():
    try:
        with open(os.path.join(APP_ROOT, 'timetable.json'), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty timetable if file not found
        return {section: {} for section in SECTIONS}

def load_daily_attendance():
    try:
        with open(os.path.join(APP_ROOT, 'daily_attendance.json'), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Initialize with empty data for each section
        daily_attendance = {section: {} for section in SECTIONS}
        with open(os.path.join(APP_ROOT, 'daily_attendance.json'), 'w') as f:
            json.dump(daily_attendance, f, indent=2)
        return daily_attendance

def save_daily_attendance(daily_attendance_data):
    with open(os.path.join(APP_ROOT, 'daily_attendance.json'), 'w') as f:
        json.dump(daily_attendance_data, f, indent=2)

def save_attendance_data(attendance_data):
    with open(os.path.join(APP_ROOT, 'attendance.json'), 'w') as f:
        json.dump(attendance_data, f, indent=2)

def create_attendance_excel(section, present_students):
    all_students = get_section_students(section)
    current_time = datetime.now()
    student_details = load_student_details()
    
    # Create a mapping of roll numbers to student details for faster lookup
    student_map = {student['rollNo']: student for student in student_details}
    
    data = []
    for student in all_students:
        student_info = student_map.get(student, {
            'name': f"Student {student}",
            'mobile': 'N/A',
            'sgpas': {}
        })
        
        data.append({
            "Roll Number": student,
            "Student Name": student_info['name'],
            "Mobile Number": student_info['mobile'],
            "SGPA Semester 1": student_info['sgpas'].get('1', 'N/A'),
            "SGPA Semester 2": student_info['sgpas'].get('2', 'N/A'),
            "Section": SECTIONS[section]["name"],
            "Status": "Present" if student in present_students else "Absent",
            "Date": current_time.strftime("%Y-%m-%d"),
            "Time": current_time.strftime("%H:%M:%S"),
            "Marked By": session.get('username', 'Unknown')
        })

    df = pd.DataFrame(data)
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance_Report')
        
        workbook = writer.book
        worksheet = writer.sheets['Attendance_Report']
        
        for col in worksheet.columns:
            max_len = max(len(str(cell.value)) for cell in col if cell.value)
            worksheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)
            
        from openpyxl.styles import Font, PatternFill, Alignment
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center")
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
    
    buffer.seek(0)
    return buffer

# login_required decorator moved to the top of the file

# ----------------- Camera Processing Class -----------------
class CameraProcessor:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)  # Reduced queue size
        self.recognition_queue = Queue(maxsize=1)  # Reduced queue size
        self.display_frame = None
        self.recognition_results = []
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.processing_thread = None
        self.recognition_thread = None
        self.lock = threading.Lock()  # Add a lock for thread safety
        
    def initialize_camera(self):
        """Initialize camera with optimized settings"""
        try:
            # Try different camera indices
            for i in range(0, 4):  # Start from 0 instead of 1
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = self.camera.read()
                    if ret:
                        logger.info(f"Camera found at index {i}")
                        break
                    else:
                        self.camera.release()
                else:
                    if i == 3:  # Last attempt
                        return False, "No camera found or cannot access camera"
            else:
                return False, "No camera found"
            
            # Optimize camera settings for performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS for better performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            return True, "Camera initialized successfully"
        except Exception as e:
            logger.error(f"Camera initialization error: {str(e)}")
            return False, f"Camera initialization error: {str(e)}"
    
    def start_processing(self, known_encodings, known_names):
        """Start camera processing threads"""
        if self.is_running:
            return
            
        self.is_running = True
        self.known_encodings = known_encodings
        self.known_names = known_names
        
        # Start frame capture thread
        self.processing_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.processing_thread.start()
        
        # Start recognition processing thread
        self.recognition_thread = threading.Thread(target=self._process_recognition, daemon=True)
        self.recognition_thread.start()
    
    def stop_processing(self):
        """Stop all processing threads"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        while not self.recognition_queue.empty():
            try:
                self.recognition_queue.get_nowait()
            except:
                break
    
    def _capture_frames(self):
        """Continuous frame capture thread"""
        while self.is_running and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.fps = self.frame_count / (current_time - self.last_fps_time)
                        self.frame_count = 0
                        self.last_fps_time = current_time
                    
                    self.frame_count += 1
                    
                    # Store frame for display
                    with self.lock:
                        self.display_frame = frame.copy()
                    
                    # Add frame to recognition queue (skip if queue is full)
                    if not self.recognition_queue.full():
                        try:
                            # Resize frame for faster processing
                            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                            self.recognition_queue.put_nowait(small_frame)
                        except:
                            pass
                else:
                    logger.warning("Failed to capture frame from camera")
                
                time.sleep(0.05)  # Slightly increased delay
            except Exception as e:
                logger.error(f"Frame capture error: {e}")
                time.sleep(1)  # Wait before trying again
    
    def _process_recognition(self):
        """Recognition processing thread"""
        global present_students
        last_recognition_time = {}
        recognition_cooldown = 2.0
        
        while self.is_running:
            try:
                frame = self.recognition_queue.get(timeout=1.0)
                recognized_faces = self._recognize_faces(frame)
                
                # Update present students with cooldown
                current_time = time.time()
                for face in recognized_faces:
                    if face['name'] != "Unknown" and face['confidence'] > 0.4:
                        last_seen = last_recognition_time.get(face['name'], 0)
                        if current_time - last_seen >= recognition_cooldown:
                            present_students.add(face['name'])
                            last_recognition_time[face['name']] = current_time
                
                self.recognition_results = recognized_faces
                time.sleep(0.1)
                
            except Exception as e:
                if "Empty" not in str(e):
                    logger.error(f"Recognition processing error: {e}")
                continue
    
    def _recognize_faces(self, frame):
        """Process face recognition on a single frame"""
        try:
            # Frame is already resized in capture thread
            rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            recognized_faces = []
            for face_encoding, face_location in zip(face_encodings, face_locations):
                name = "Unknown"
                confidence = 0

                if self.known_encodings and len(self.known_encodings) > 0:
                    matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.4)
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                            name = self.known_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]

                # Scale back face locations to original frame size
                top, right, bottom, left = [coord * 2 for coord in face_location]
                
                recognized_faces.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (int(top), int(right), int(bottom), int(left))
                })

            return recognized_faces
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
            return []
    
    def get_display_frame_with_boxes(self):
        """Get current display frame with recognition boxes"""
        with self.lock:
            if self.display_frame is None:
                # Return a black frame if no camera feed
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No camera feed", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return placeholder
                
            frame = self.display_frame.copy()
        
        for face in self.recognition_results:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']

            if name == "Unknown":
                color = (0, 0, 255)
                bg_color = (0, 0, 200)
                label = "Unknown"
            else:
                color = (0, 255, 0)
                bg_color = (0, 200, 0)
                label = name

            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = top - 15 if top - 15 > label_height else bottom + label_height + 15
            
            cv2.rectangle(frame, 
                         (left, label_y - label_height - 15), 
                         (left + label_width + 20, label_y + 5), 
                         bg_color, cv2.FILLED)
            
            cv2.putText(frame, label, 
                       (left + 10, label_y - 8), 
                       font, font_scale, (255, 255, 255), thickness)
            
            if name != "Unknown" and confidence > 0:
                confidence_text = f"Conf: {confidence:.1%}"
                conf_width, conf_height = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, 
                             (left, label_y + 10), 
                             (left + conf_width + 10, label_y + conf_height + 20), 
                             bg_color, cv2.FILLED)
                cv2.putText(frame, confidence_text, 
                           (left + 5, label_y + conf_height + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, fps_text, (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        return frame

# ----------------- Routes -----------------
@app.route('/')
def index():
    if 'authenticated' in session and session['authenticated']:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password are required'})
        
        user_data = authenticate_user(username, password)
        if user_data:
            session['authenticated'] = True
            session['username'] = username
            session['faculty_name'] = user_data.get('name', username)
            session['sections'] = user_data.get('sections', list(SECTIONS.keys()))
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})
    
    # GET request - show login page
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    global camera_processor, attendance_started, present_students
    
    if camera_processor:
        camera_processor.stop_processing()
        camera_processor = None
    
    attendance_started = False
    present_students = set()
    
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get only sections assigned to this teacher
    user_sections = session.get('sections', [])
    filtered_sections = {k: v for k, v in SECTIONS.items() if k in user_sections}
    
    # Get attendance statistics for graph display
    section_stats = {}
    for section_id, section_info in filtered_sections.items():
        students = get_section_students(section_id)
        total_students = len(students)
        section_stats[section_id] = {
            'name': section_info['name'],
            'total': total_students,
            'present': 0,  # Will be updated when attendance is taken
            'absent': total_students  # Initially all are absent
        }
    
    # Load student data for graphs
    students_data = load_students_data()
    
    # Debug log
    print(f"User sections: {user_sections}")
    print(f"Filtered sections: {filtered_sections}")
    
    return render_template('dashboard.html', 
                         username=session.get('username'),
                         students_data=students_data,
                         faculty_name=session.get('faculty_name', session.get('username')),
                         sections=filtered_sections,  # Pass only filtered sections
                         user_sections=user_sections,  # Pass user sections for debugging
                         section_stats=section_stats)

@app.route('/api/sections')
@login_required
def get_sections():
    sections_data = []
    for key, value in SECTIONS.items():
        # Get attendance statistics for this section
        students = get_section_students(key)
        present_count = sum(1 for student in students if student in present_students)
        total_count = len(students)
        attendance_rate = (present_count / total_count) * 100 if total_count > 0 else 0
        
        sections_data.append({
            'id': key,
            'name': value['name'],
            'start': value['start'],
            'present': present_count,
            'absent': total_count - present_count,
            'attendance_rate': round(attendance_rate, 1),
            'end': value['end'],
            'prefix': value['prefix'],
            'total_students': value['end'] - value['start'] + 1
        })
    return jsonify(sections_data)

@app.route('/api/timetable/<section_id>')
@login_required
def get_section_timetable(section_id):
    """Get timetable for a specific section"""
    if section_id not in SECTIONS:
        return jsonify({
            'success': False,
            'message': 'Section not found'
        })
    
    timetable = load_timetable()
    if section_id in timetable:
        return jsonify({
            'success': True,
            'timetable': timetable[section_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Timetable not found for this section'
        })

@app.route('/api/start_attendance', methods=['POST'])
@login_required
def start_attendance():
    global camera_processor, attendance_started
    
    section = request.json.get('section')
    if not section:
        return jsonify({'success': False, 'message': 'No section provided'})
    
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        return jsonify({'success': False, 'message': 'No encodings found'})
    
    section_config = SECTIONS[section]
    section_encodings = []
    section_names = []
    
    for enc, name in zip(known_encodings, known_names):
        if name.startswith(section_config["prefix"]):
            try:
                roll_num = int(name[len(section_config["prefix"]):])
                if section_config["start"] <= roll_num <= section_config["end"]:
                    section_encodings.append(enc)
                    section_names.append(name)
            except ValueError:
                continue
    
    if not section_encodings:
        return jsonify({'success': False, 'message': f'No encodings found for {SECTIONS[section]["name"]}'})
    
    camera_processor = CameraProcessor()
    success, message = camera_processor.initialize_camera()
    
    if not success:
        return jsonify({'success': False, 'message': message})
    
    camera_processor.start_processing(section_encodings, section_names)
    attendance_started = True
    
    return jsonify({'success': True, 'message': 'Attendance session started'})

@app.route('/api/stop_attendance', methods=['POST'])
@login_required
def stop_attendance():
    global camera_processor, attendance_started
    
    if camera_processor:
        camera_processor.stop_processing()
        camera_processor = None
    
    # Save attendance data to JSON file
    section = request.json.get('section')
    if section and section in SECTIONS:
        # Check if user has access to this section
        user_sections = session.get('sections', [])
        if section not in user_sections:
            return jsonify({'error': 'You do not have access to this section'}), 403
            
        date_str = datetime.now().strftime('%Y-%m-%d')
        attendance_data = load_attendance_data()
        
        if section not in attendance_data:
            attendance_data[section] = {}
            
        if date_str not in attendance_data[section]:
            attendance_data[section][date_str] = {}
        
        # Convert present_students set to a dictionary with 1 for present
        all_students = get_section_students(section)
        for student in all_students:
            attendance_data[section][date_str][student] = 1 if student in present_students else 0
        
        save_attendance_data(attendance_data)
    
    attendance_started = False
    return jsonify({'success': True, 'message': 'Attendance session stopped and data saved'})

@app.route('/api/reset_attendance', methods=['POST'])
@login_required
def reset_attendance():
    global present_students
    
    present_students = set()
    return jsonify({'success': True, 'message': 'Attendance data reset'})

@app.route('/api/add_student', methods=['POST'])
@login_required
def add_student():
    global present_students
    
    student = request.json.get('student')
    if student:
        present_students.add(student)
        return jsonify({'success': True, 'message': f'{student} marked present'})
    
    return jsonify({'success': False, 'message': 'No student provided'})

@app.route('/api/update_daily_attendance', methods=['POST'])
@login_required
def update_daily_attendance():
    """Update daily attendance for a student"""
    data = request.json
    roll_number = data.get('roll_number')
    date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    attended = data.get('attended', 0)
    total = data.get('total', 0)
    
    if not roll_number:
        return jsonify({'success': False, 'message': 'No roll number provided'})
    
    # Find which section this student belongs to
    student_section = None
    for section_id, section_config in SECTIONS.items():
        prefix = section_config["prefix"]
        try:
            roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
            if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                student_section = section_id
                break
        except (ValueError, TypeError):
            continue
    
    if not student_section:
        return jsonify({'success': False, 'message': 'Student section not found'})
    
    # Load daily attendance data
    daily_attendance = load_daily_attendance()
    
    # Initialize section data if not exists
    if student_section not in daily_attendance:
        daily_attendance[student_section] = {}
    
    # Initialize date data if not exists
    if date not in daily_attendance[student_section]:
        daily_attendance[student_section][date] = {}
    
    # Update student attendance
    daily_attendance[student_section][date][roll_number] = {
        'attended': attended,
        'total': total
    }
    
    # Save updated data
    save_daily_attendance(daily_attendance)
    
    return jsonify({
        'success': True,
        'message': f'Daily attendance updated for {roll_number}',
        'attendance_percentage': calculate_attendance_percentage(roll_number)
    })

@app.route('/api/student/<roll_number>')
@login_required
def get_student_data(roll_number):
    try:
        students_data = load_students_data()
        
        # Get attendance history for this student
        attendance_history = get_student_attendance_history(roll_number)
        
        # Get daily attendance data
        daily_attendance = get_student_daily_attendance(roll_number)
        
        # Calculate attendance percentage
        attendance_percentage = calculate_attendance_percentage(roll_number)
        
        # Find which section this student belongs to
        student_section = None
        for section_id, section_config in SECTIONS.items():
            prefix = section_config["prefix"]
            try:
                roll_num = int(roll_number[len(prefix):]) if roll_number.startswith(prefix) else None
                if roll_num and section_config["start"] <= roll_num <= section_config["end"]:
                    student_section = section_id
                    break
            except (ValueError, TypeError):
                continue
        
        # Process daily attendance to include subject information
        for day in daily_attendance:
            day_name = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%A")
            if student_section and day_name in TIMETABLE.get(student_section, {}):
                subjects_list = TIMETABLE[student_section][day_name]
                # Generate random attendance for each subject if not already present
                if "subjects" not in day or not day["subjects"]:
                    import random
                    attended_count = day["classes_attended"]
                    subjects = []
                    
                    # Randomly distribute attendance across subjects
                    attended_subjects = random.sample(subjects_list, attended_count) if attended_count <= len(subjects_list) else subjects_list
                    
                    for subject in subjects_list:
                        subjects.append({
                            "name": subject,
                            "attended": subject in attended_subjects
                        })
                    day["subjects"] = subjects
        
        if roll_number in students_data:
            student = students_data[roll_number]
            # Calculate CGPA
            student['cgpa'] = calculate_cgpa(student.get('sgpas', {}))
            # Add image URL
            student['image_url'] = get_student_image_url(roll_number)
            # Add attendance history
            student['attendance_history'] = attendance_history
            # Add daily attendance data
            student['daily_attendance'] = daily_attendance
            # Add attendance percentage
            student['attendance_percentage'] = attendance_percentage
            
            # Generate SGPA graph if Plotly is available
            if PLOTLY_AVAILABLE and student.get('sgpas'):
                student['sgpa_graph'] = create_enhanced_sgpa_graph(student.get('sgpas', {}))
            
            return jsonify({
                'success': True,
                'student': student
            })
        else:
            # Try to find student in details.json
            student_details = load_student_details()
            for student in student_details:
                if student.get('rollNo') == roll_number:
                    student['cgpa'] = calculate_cgpa(student.get('sgpas', {}))
                    # Add image URL
                    student['image_url'] = get_student_image_url(roll_number)
                    # Add attendance history
                    student['attendance_history'] = attendance_history
                    # Add daily attendance data
                    student['daily_attendance'] = daily_attendance
                    # Add attendance percentage
                    student['attendance_percentage'] = attendance_percentage
                    
                    # Generate SGPA graph if Plotly is available
                    if PLOTLY_AVAILABLE and student.get('sgpas'):
                        student['sgpa_graph'] = create_enhanced_sgpa_graph(student.get('sgpas', {}))
                    
                    return jsonify({
                        'success': True,
                        'student': student
                    })
            
            # If not found, create sample data
            sample_student = {
                'rollNo': roll_number,
                'name': f'Student {roll_number}',
                'mobile': f'9{np.random.randint(100000000, 999999999)}',
                'sgpas': {
                    '1': str(round(np.random.uniform(7.0, 9.5), 2)),
                    '2': str(round(np.random.uniform(7.0, 9.5), 2)),
                    '3': str(round(np.random.uniform(7.0, 9.5), 2)),
                    '4': str(round(np.random.uniform(7.0, 9.5), 2))
                }
            }
            sample_student['cgpa'] = calculate_cgpa(sample_student['sgpas'])
            # Add image URL
            sample_student['image_url'] = get_student_image_url(roll_number)
            # Add attendance history
            sample_student['attendance_history'] = attendance_history
            # Add daily attendance data
            sample_student['daily_attendance'] = daily_attendance
            # Add attendance percentage
            sample_student['attendance_percentage'] = attendance_percentage
            
            # Generate SGPA graph if Plotly is available
            if PLOTLY_AVAILABLE:
                sample_student['sgpa_graph'] = create_enhanced_sgpa_graph(sample_student['sgpas'])
            
            return jsonify({
                'success': True,
                'student': sample_student
            })
    except Exception as e:
        print(f"Error fetching student data: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Error fetching student data: {str(e)}"
        })

@app.route('/api/attendance_data')
@login_required
def get_attendance_data():
    section = request.args.get('section')
    if not section or section not in SECTIONS:
        return jsonify({'error': 'Invalid section'}), 400
    
    all_students = get_section_students(section)
    student_details = load_student_details()
    student_map = {student['rollNo']: student for student in student_details}
    
    attendance_data = []
    
    for i, student in enumerate(all_students, 1):
        student_info = student_map.get(student, {
            'name': f"Student {student}",
            'mobile': 'N/A'
        })
        
        status = "Present" if student in present_students else "Absent"
        attendance_data.append({
            "sno": i,
            "roll_number": student,
            "student_name": student_info['name'],
            "mobile": student_info['mobile'],
            "status": status,
            "section": SECTIONS[section]["name"]
        })
    
    present_count = len(present_students)
    total_count = len(all_students)
    attendance_rate = (present_count / total_count) * 100 if total_count > 0 else 0
    
    return jsonify({
        'data': attendance_data,
        'summary': {
            'present': present_count,
            'total': total_count,
            'rate': round(attendance_rate, 1)
        }
    })

@app.route('/api/student_details/<roll_number>')
@login_required
def get_student_details(roll_number):
    """Get detailed information for a specific student"""
    student_info = get_student_info(roll_number)
    return jsonify(student_info)

@app.route('/api/teacher_profile')
@login_required
def teacher_profile():
    """Get attendance statistics for the teacher's profile"""
    username = session.get('username')
    faculty_name = session.get('faculty_name')
    user_sections = session.get('sections', [])
    
    # Load attendance data
    attendance_data = load_attendance_data()
    
    # Calculate statistics for each section
    section_stats = {}
    total_classes = 0
    total_present = 0
    total_absent = 0
    
    for section_id in user_sections:
        if section_id in attendance_data:
            section_dates = attendance_data[section_id]
            section_classes = len(section_dates)
            total_classes += section_classes
            
            section_present = 0
            section_absent = 0
            students = get_section_students(section_id)
            total_students = len(students)
            
            # Calculate daily attendance for the section
            daily_attendance = []
            for date, students_status in section_dates.items():
                day_present = sum(1 for status in students_status.values() if status == 1)
                day_absent = total_students - day_present
                attendance_rate = round((day_present / total_students) * 100, 2) if total_students > 0 else 0
                daily_attendance.append({
                    'date': date,
                    'present': day_present,
                    'absent': day_absent,
                    'rate': attendance_rate,
                    'attendance_rate': attendance_rate  # For backward compatibility
                })
                section_present += day_present
                section_absent += day_absent
            
            total_present += section_present
            total_absent += section_absent
            
            # Calculate average attendance rate for the section
            avg_attendance_rate = round((section_present / (section_present + section_absent)) * 100, 2) if (section_present + section_absent) > 0 else 0
            
            # Calculate weekly trends
            sorted_attendance = sorted(daily_attendance, key=lambda x: x['date'])
            weekly_trends = []
            if sorted_attendance:
                # Group by week
                from datetime import datetime, timedelta
                week_data = {}
                for day in sorted_attendance:
                    date_obj = datetime.strptime(day['date'], '%Y-%m-%d')
                    week_start = (date_obj - timedelta(days=date_obj.weekday())).strftime('%Y-%m-%d')
                    if week_start not in week_data:
                        week_data[week_start] = {'present': 0, 'absent': 0, 'total': 0}
                    week_data[week_start]['present'] += day['present']
                    week_data[week_start]['absent'] += day['absent']
                    week_data[week_start]['total'] += 1
                
                # Calculate weekly averages
                for week_start, data in week_data.items():
                    week_rate = round((data['present'] / (data['present'] + data['absent'])) * 100, 2) if (data['present'] + data['absent']) > 0 else 0
                    weekly_trends.append({
                        'week': week_start,
                        'rate': week_rate,
                        'classes': data['total']
                    })
            
            section_stats[section_id] = {
                'name': SECTIONS[section_id]['name'],
                'total_classes': section_classes,
                'total_students': total_students,
                'present_count': section_present,
                'absent_count': section_absent,
                'avg_attendance_rate': avg_attendance_rate,
                'daily_attendance': sorted(daily_attendance, key=lambda x: x['date'], reverse=True),
                'weekly_trends': sorted(weekly_trends, key=lambda x: x['week'])
            }
        else:
            # Add empty stats for sections with no attendance data
            section_stats[section_id] = {
                'name': SECTIONS[section_id]['name'],
                'total_classes': 0,
                'total_students': len(get_section_students(section_id)),
                'present_count': 0,
                'absent_count': 0,
                'avg_attendance_rate': 0,
                'daily_attendance': [],
                'weekly_trends': []
            }
    
    # Calculate overall attendance rate
    overall_rate = round((total_present / (total_present + total_absent)) * 100, 2) if (total_present + total_absent) > 0 else 0
    
    return jsonify({
        'username': username,
        'faculty_name': faculty_name,
        'total_classes': total_classes,
        'total_present': total_present,
        'total_absent': total_absent,
        'overall_attendance_rate': overall_rate,
        'section_stats': section_stats
    })

@app.route('/api/export_excel')
@login_required
def export_excel():
    section = request.args.get('section')
    if not section:
        return jsonify({'error': 'No section provided'})
    
    buffer = create_attendance_excel(section, present_students)
    filename = f"Attendance_{SECTIONS[section]['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/api/save_manual_attendance', methods=['POST'])
@login_required
def save_manual_attendance():
    data = request.json
    section = data.get('section')
    attendance = data.get('attendance')
    
    if not section or not attendance:
        return jsonify({'success': False, 'message': 'Section and attendance data are required'})
    
    # Check if user has access to this section
    user_sections = session.get('sections', [])
    if section not in user_sections:
        return jsonify({'success': False, 'message': 'You do not have access to this section'})
    
    # Load current attendance data
    attendance_data = load_attendance_data()
    date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize section and date if they don't exist
    if section not in attendance_data:
        attendance_data[section] = {}
    if date_str not in attendance_data[section]:
        attendance_data[section][date_str] = {}
    
    # Update attendance data with manual entries
    for entry in attendance:
        roll_number = entry.get('roll_number')
        status = 1 if entry.get('status') == 'Present' else 0
        attendance_data[section][date_str][roll_number] = status
    
    # Save updated attendance data
    save_attendance_data(attendance_data)
    
    # Update present_students set to match the manual attendance
    global present_students
    present_students = set()
    for entry in attendance:
        if entry.get('status') == 'Present':
            present_students.add(entry.get('roll_number'))
    
    return jsonify({'success': True, 'message': 'Attendance saved successfully'})

@app.route('/api/export_csv')
@login_required
def export_csv():
    section = request.args.get('section')
    if not section:
        return jsonify({'error': 'No section provided'})
    
    all_students = get_section_students(section)
    student_details = load_student_details()
    student_map = {student['rollNo']: student for student in student_details}
    
    attendance_data = []
    
    for i, student in enumerate(all_students, 1):
        student_info = student_map.get(student, {
            'name': f"Student {student}",
            'mobile': 'N/A'
        })
        
        status = "Present" if student in present_students else "Absent"
        attendance_data.append({
            "S.No": i,
            "Roll Number": student,
            "Student Name": student_info['name'],
            "Mobile": student_info['mobile'],
            "Status": status,
            "Section": SECTIONS[section]["name"]
        })
    
    df = pd.DataFrame(attendance_data)
    csv_data = df.to_csv(index=False)
    
    filename = f"Attendance_{SECTIONS[section]['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

@app.route('/video_feed')
@login_required
def video_feed():
    def generate():
        global camera_processor
        
        while True:
            try:
                if camera_processor and camera_processor.is_running:
                    frame_with_boxes = camera_processor.get_display_frame_with_boxes()
                    
                    # Encode frame as JPEG with lower quality for better performance
                    ret, jpeg = cv2.imencode('.jpg', frame_with_boxes, 
                                            [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                    else:
                        # Create a placeholder frame
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, "Encoding error", (150, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        ret, jpeg = cv2.imencode('.jpg', placeholder)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                else:
                    # Create a placeholder frame when camera is not running
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Camera not active", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    ret, jpeg = cv2.imencode('.jpg', placeholder)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                
                time.sleep(0.05)  # Control frame rate
            except Exception as e:
                logger.error(f"Video feed error: {e}")
                # Create an error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Error: {str(e)}", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/recognition_status')
@login_required
def recognition_status():
    global camera_processor, present_students
    
    if camera_processor:
        return jsonify({
            'fps': round(camera_processor.fps, 1),
            'faces_detected': len(camera_processor.recognition_results),
            'present_count': len(present_students),
            'recognition_results': camera_processor.recognition_results,
            'present_students': list(present_students)[-8:] if present_students else []
        })
    
    return jsonify({
        'fps': 0,
        'faces_detected': 0,
        'present_count': len(present_students),
        'recognition_results': [],
        'present_students': list(present_students)[-8:] if present_students else []
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)