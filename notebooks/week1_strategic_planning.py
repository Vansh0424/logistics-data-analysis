# Week 1 - Strategic Planning and Data Exploration
# Logistics Data Analysis and Predictive Optimization

import pandas as pd
import numpy as np

# Project objectives
objectives = [
    "Measure logistics performance using KPIs",
    "Identify factors associated with delivery delays",
    "Prepare logistics data for analysis",
    "Perform exploratory data analysis",
    "Develop predictive models",
    "Propose logistics optimization strategies"
]

print("LOGISTICS DATA ANALYSIS PROJECT")
print("=" * 40)

print("\nProject Objectives:")
for number, objective in enumerate(objectives, 1):
    print(f"{number}. {objective}")

# Planned KPIs
kpis = {
    "On-Time Delivery Rate": "Percentage of shipments delivered on time",
    "Late Delivery Rate": "Percentage of shipments classified as late",
    "Average Delivery Delay": "Difference between actual and scheduled shipping days",
    "Average Shipping Cost": "Average logistics cost per order",
    "Order Profitability": "Profit generated from logistics operations"
}

print("\nKey Performance Indicators:")
for kpi, definition in kpis.items():
    print(f"- {kpi}: {definition}")

print("\nPlanned Workflow:")
workflow = [
    "Data Collection",
    "Data Cleaning",
    "Feature Preparation",
    "Exploratory Data Analysis",
    "Data Visualization",
    "Predictive Modeling",
    "Optimization",
    "Final Reporting"
]

for step, activity in enumerate(workflow, 1):
    print(f"{step}. {activity}")

print("\nWeek 1 planning completed successfully.")
