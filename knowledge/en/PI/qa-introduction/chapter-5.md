---
title: "Chapter 5: Quality Control Automation with Python"
chapter_title: "Chapter 5: Quality Control Automation with Python"
subtitle: Real-time monitoring, dashboards, and machine learning predictions
---

[AI Terakoya Top](<../index.html>)›[Process Informatics](<../../index.html>)›[QA](<../../PI/qa-introduction/index.html>)›Chapter 5

🌐 EN | [🇯🇵 JP](<../../../jp/PI/qa-introduction/chapter-5.html>) | Last sync: 2025-11-16

### What You'll Learn in This Chapter

  * Building real-time SPC monitoring and alert systems
  * Creating interactive quality dashboards with Plotly
  * Automated report generation (PDF/Excel) and email delivery
  * Quality data management through database integration
  * Defect prediction systems using machine learning
  * Data collection automation through external API integration
  * Complete quality management system workflow

## 5.1 Real-time Quality Monitoring System

### 5.1.1 Automated SPC Monitoring and Alert System

Build a system that monitors processes in real-time on the manufacturing floor and automatically generates alerts when control limits are exceeded.

Example 1: Real-time SPC Monitoring System
    
    
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from typing import List, Dict, Optional
    import json
    
    class RealTimeSPCMonitor:
        """Real-time SPC Monitoring System
    
        Monitors manufacturing processes in real-time and
        automatically generates alerts when control limits are exceeded.
        Also detects abnormal patterns.
        """
    
        def __init__(self, process_name: str, target: float,
                     ucl: float, lcl: float,
                     email_recipients: List[str] = None):
            """
            Args:
                process_name: Process name
                target: Target value
                ucl: Upper control limit
                lcl: Lower control limit
                email_recipients: List of email addresses for alerts
            """
            self.process_name = process_name
            self.target = target
            self.ucl = ucl
            self.lcl = lcl
            self.email_recipients = email_recipients or []
    
            self.data_points = []
            self.alerts = []
    
            # Rule settings for abnormal pattern detection (Western Electric Rules)
            self.rules = {
                "rule1": "1 point beyond control limits",
                "rule2": "2 of 3 consecutive points beyond 2-sigma (same side)",
                "rule3": "4 of 5 consecutive points beyond 1-sigma (same side)",
                "rule4": "8 consecutive points on same side",
                "rule5": "6 consecutive points monotonically increasing or decreasing",
                "rule6": "14 consecutive points alternating up and down"
            }
    
            # Calculate sigma (standard deviation)
            self.sigma = (ucl - target) / 3
    
        def add_measurement(self, value: float, timestamp: datetime = None) -> Dict:
            """Add measurement value and perform real-time evaluation
    
            Args:
                value: Measurement value
                timestamp: Measurement time (defaults to current time)
    
            Returns:
                Evaluation result
            """
            if timestamp is None:
                timestamp = datetime.now()
    
            data_point = {
                "timestamp": timestamp,
                "value": value,
                "in_control": self.lcl <= value <= self.ucl
            }
    
            self.data_points.append(data_point)
    
            # Anomaly detection
            violations = self._check_violations()
    
            if violations:
                alert = self._generate_alert(data_point, violations)
                self.alerts.append(alert)
    
                # Send email
                if self.email_recipients:
                    self._send_alert_email(alert)
    
                return {
                    "status": "ALERT",
                    "value": value,
                    "violations": violations,
                    "alert_id": alert["alert_id"]
                }
            else:
                return {
                    "status": "OK",
                    "value": value,
                    "violations": []
                }
    
        def _check_violations(self) -> List[str]:
            """Detect anomalies based on Western Electric Rules
    
            Returns:
                List of violated rules
            """
            violations = []
    
            if len(self.data_points) == 0:
                return violations
    
            recent_data = self.data_points[-14:]  # Evaluate latest 14 points
    
            # Rule 1: 1 point beyond control limits
            if not recent_data[-1]["in_control"]:
                violations.append(self.rules["rule1"])
    
            if len(recent_data) < 3:
                return violations
    
            # Rule 2: 2 of 3 consecutive points beyond 2-sigma (same side)
            last_3 = recent_data[-3:]
            values_3 = [p["value"] for p in last_3]
    
            upper_2sigma = self.target + 2 * self.sigma
            lower_2sigma = self.target - 2 * self.sigma
    
            upper_violations = sum(1 for v in values_3 if v > upper_2sigma)
            lower_violations = sum(1 for v in values_3 if v < lower_2sigma)
    
            if upper_violations >= 2 or lower_violations >= 2:
                violations.append(self.rules["rule2"])
    
            if len(recent_data) < 5:
                return violations
    
            # Rule 3: 4 of 5 consecutive points beyond 1-sigma (same side)
            last_5 = recent_data[-5:]
            values_5 = [p["value"] for p in last_5]
    
            upper_1sigma = self.target + self.sigma
            lower_1sigma = self.target - self.sigma
    
            upper_violations_5 = sum(1 for v in values_5 if v > upper_1sigma)
            lower_violations_5 = sum(1 for v in values_5 if v < lower_1sigma)
    
            if upper_violations_5 >= 4 or lower_violations_5 >= 4:
                violations.append(self.rules["rule3"])
    
            if len(recent_data) < 8:
                return violations
    
            # Rule 4: 8 consecutive points on same side
            last_8 = recent_data[-8:]
            values_8 = [p["value"] for p in last_8]
    
            all_above = all(v > self.target for v in values_8)
            all_below = all(v < self.target for v in values_8)
    
            if all_above or all_below:
                violations.append(self.rules["rule4"])
    
            # Rule 5: 6 consecutive points monotonically increasing or decreasing
            if len(recent_data) >= 6:
                last_6 = recent_data[-6:]
                values_6 = [p["value"] for p in last_6]
    
                increasing = all(values_6[i] < values_6[i+1] for i in range(5))
                decreasing = all(values_6[i] > values_6[i+1] for i in range(5))
    
                if increasing or decreasing:
                    violations.append(self.rules["rule5"])
    
            # Rule 6: 14 consecutive points alternating up and down
            if len(recent_data) >= 14:
                values_14 = [p["value"] for p in recent_data]
    
                alternating = all(
                    (values_14[i] - values_14[i-1]) * (values_14[i+1] - values_14[i]) < 0
                    for i in range(1, 13)
                )
    
                if alternating:
                    violations.append(self.rules["rule6"])
    
            return violations
    
        def _generate_alert(self, data_point: Dict, violations: List[str]) -> Dict:
            """Generate alert
    
            Args:
                data_point: Data point
                violations: List of violated rules
    
            Returns:
                Alert information
            """
            alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
            alert = {
                "alert_id": alert_id,
                "timestamp": data_point["timestamp"],
                "process_name": self.process_name,
                "value": data_point["value"],
                "target": self.target,
                "ucl": self.ucl,
                "lcl": self.lcl,
                "violations": violations,
                "severity": "Critical" if self.rules["rule1"] in violations else "Warning",
                "acknowledged": False
            }
    
            return alert
    
        def _send_alert_email(self, alert: Dict):
            """Send alert email
    
            Args:
                alert: Alert information
            """
            # Actual implementation requires SMTP server configuration
            # Here we only output to log
            print(f"\n{'='*60}")
            print(f"📧 Sending Alert Email")
            print(f"{'='*60}")
            print(f"To: {', '.join(self.email_recipients)}")
            print(f"Subject: [{alert['severity']}] {self.process_name} Process Anomaly Detected")
            print(f"\n--- Email Body ---")
            print(f"Alert ID: {alert['alert_id']}")
            print(f"Process: {self.process_name}")
            print(f"Occurrence Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Measurement Value: {alert['value']:.3f}")
            print(f"Target Value: {self.target:.3f}")
            print(f"Control Limits: {self.lcl:.3f} - {self.ucl:.3f}")
            print(f"\nDetected Abnormal Patterns:")
            for violation in alert['violations']:
                print(f"  • {violation}")
            print(f"\nPlease check the process immediately and take action as needed.")
            print(f"{'='*60}\n")
    
        def get_control_chart_data(self) -> pd.DataFrame:
            """Get control chart data
    
            Returns:
                DataFrame for control chart plotting
            """
            if not self.data_points:
                return pd.DataFrame()
    
            df = pd.DataFrame(self.data_points)
    
            # Add control limit lines
            df['target'] = self.target
            df['ucl'] = self.ucl
            df['lcl'] = self.lcl
            df['upper_2sigma'] = self.target + 2 * self.sigma
            df['lower_2sigma'] = self.target - 2 * self.sigma
            df['upper_1sigma'] = self.target + self.sigma
            df['lower_1sigma'] = self.target - self.sigma
    
            return df
    
        def get_alert_summary(self) -> Dict:
            """Get alert summary
    
            Returns:
                Alert statistics
            """
            if not self.alerts:
                return {"total_alerts": 0}
    
            df = pd.DataFrame(self.alerts)
    
            summary = {
                "total_alerts": len(self.alerts),
                "critical_alerts": (df['severity'] == 'Critical').sum(),
                "warning_alerts": (df['severity'] == 'Warning').sum(),
                "latest_alert": self.alerts[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "most_common_violation": df['violations'].explode().mode()[0]
                                         if len(df) > 0 else None
            }
    
            return summary
    
    
    # Usage example
    # Start real-time SPC monitoring system
    monitor = RealTimeSPCMonitor(
        process_name="Molding Temperature",
        target=200.0,
        ucl=210.0,
        lcl=190.0,
        email_recipients=["quality@example.com", "production@example.com"]
    )
    
    # Simulation: Add consecutive measurements
    np.random.seed(42)
    
    print("=== Real-time SPC Monitoring Started ===\n")
    
    # Normal data
    for i in range(10):
        value = np.random.normal(200.0, 2.5)
        result = monitor.add_measurement(value, datetime.now() + timedelta(minutes=i))
        print(f"[{i+1:2d}] Measurement: {value:.2f} → {result['status']}")
    
    # Anomaly occurrence: Control limit exceeded
    for i in range(10, 13):
        value = np.random.normal(212.0, 1.0)  # Exceeds UCL
        result = monitor.add_measurement(value, datetime.now() + timedelta(minutes=i))
        print(f"[{i+1:2d}] Measurement: {value:.2f} → {result['status']}")
        if result['status'] == 'ALERT':
            print(f"     ⚠️  Alert Generated: {result['alert_id']}")
    
    # Display alert summary
    print("\n=== Alert Summary ===")
    summary = monitor.get_alert_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

#### 💡 Implementation Points

  * **Western Electric Rules** : Automatically detects 6 abnormal patterns
  * **Real-time Evaluation** : Immediate determination upon measurement value addition
  * **Alert Severity Classification** : Critical (beyond control limits) and Warning (pattern anomaly)
  * **Notification Feature** : Extensible to email, Slack, SMS, etc.

### 5.1.2 Interactive Quality Dashboard (Plotly)

Build an interactive quality dashboard that updates in real-time using Plotly.

Example 2: Plotly Interactive Dashboard
    
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    class QualityDashboard:
        """Interactive Quality Dashboard
    
        Visualize real-time quality metrics using Plotly.
        Integrates control charts, histograms, Pareto charts, and time series trends.
        """
    
        def __init__(self):
            self.data = {
                "control_chart": [],
                "defect_data": [],
                "kpi_data": []
            }
    
        def add_control_chart_data(self, timestamp: datetime, value: float,
                                  target: float, ucl: float, lcl: float):
            """Add control chart data
    
            Args:
                timestamp: Time
                value: Measurement value
                target: Target value
                ucl: Upper control limit
                lcl: Lower control limit
            """
            self.data["control_chart"].append({
                "timestamp": timestamp,
                "value": value,
                "target": target,
                "ucl": ucl,
                "lcl": lcl
            })
    
        def add_defect_data(self, defect_type: str, count: int):
            """Add defect data
    
            Args:
                defect_type: Defect type
                count: Count
            """
            self.data["defect_data"].append({
                "type": defect_type,
                "count": count
            })
    
        def add_kpi_data(self, kpi_name: str, actual: float,
                        target: float, unit: str):
            """Add KPI data
    
            Args:
                kpi_name: KPI name
                actual: Actual value
                target: Target value
                unit: Unit
            """
            self.data["kpi_data"].append({
                "kpi": kpi_name,
                "actual": actual,
                "target": target,
                "unit": unit,
                "achievement": (actual / target * 100) if target != 0 else 0
            })
    
        def create_dashboard(self, title: str = "Quality Dashboard") -> go.Figure:
            """Generate dashboard
    
            Args:
                title: Dashboard title
    
            Returns:
                Plotly Figure object
            """
            # Create 2x2 subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Xbar Control Chart",
                    "Defect Pareto Chart",
                    "KPI Achievement Status",
                    "Time Series Trend (7 days)"
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
    
            # 1. Control chart (top left)
            if self.data["control_chart"]:
                self._add_control_chart(fig, row=1, col=1)
    
            # 2. Pareto chart (top right)
            if self.data["defect_data"]:
                self._add_pareto_chart(fig, row=1, col=2)
    
            # 3. KPI achievement status (bottom left)
            if self.data["kpi_data"]:
                self._add_kpi_chart(fig, row=2, col=1)
    
            # 4. Time series trend (bottom right)
            if self.data["control_chart"]:
                self._add_trend_chart(fig, row=2, col=2)
    
            # Layout settings
            fig.update_layout(
                title_text=title,
                title_font_size=20,
                showlegend=True,
                height=800,
                template="plotly_white"
            )
    
            return fig
    
        def _add_control_chart(self, fig: go.Figure, row: int, col: int):
            """Add control chart (internal use)"""
            df = pd.DataFrame(self.data["control_chart"])
    
            # Measurement values
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['value'],
                    mode='lines+markers',
                    name='Measurement',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=col
            )
    
            # Target value
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['target'],
                    mode='lines',
                    name='Target',
                    line=dict(color='green', width=2, dash='dash')
                ),
                row=row, col=col
            )
    
            # UCL
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['ucl'],
                    mode='lines',
                    name='UCL',
                    line=dict(color='red', width=1, dash='dot')
                ),
                row=row, col=col
            )
    
            # LCL
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['lcl'],
                    mode='lines',
                    name='LCL',
                    line=dict(color='red', width=1, dash='dot')
                ),
                row=row, col=col
            )
    
            fig.update_xaxes(title_text="Time", row=row, col=col)
            fig.update_yaxes(title_text="Measurement", row=row, col=col)
    
        def _add_pareto_chart(self, fig: go.Figure, row: int, col: int):
            """Add Pareto chart (internal use)"""
            df = pd.DataFrame(self.data["defect_data"])
            df = df.sort_values('count', ascending=False)
    
            # Calculate cumulative ratio
            df['cumulative'] = df['count'].cumsum()
            df['cumulative_pct'] = df['cumulative'] / df['count'].sum() * 100
    
            # Bar chart
            fig.add_trace(
                go.Bar(
                    x=df['type'],
                    y=df['count'],
                    name='Defect Count',
                    marker_color='lightblue',
                    yaxis='y'
                ),
                row=row, col=col
            )
    
            # Cumulative line chart
            fig.add_trace(
                go.Scatter(
                    x=df['type'],
                    y=df['cumulative_pct'],
                    name='Cumulative %',
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=8),
                    yaxis='y2'
                ),
                row=row, col=col
            )
    
            fig.update_xaxes(title_text="Defect Type", row=row, col=col)
            fig.update_yaxes(title_text="Count", row=row, col=col)
    
        def _add_kpi_chart(self, fig: go.Figure, row: int, col: int):
            """Add KPI achievement status (internal use)"""
            df = pd.DataFrame(self.data["kpi_data"])
    
            # Color coding (based on achievement rate)
            colors = ['green' if ach >= 100 else 'orange' if ach >= 80 else 'red'
                     for ach in df['achievement']]
    
            fig.add_trace(
                go.Bar(
                    x=df['kpi'],
                    y=df['achievement'],
                    name='Achievement Rate',
                    marker_color=colors,
                    text=df['achievement'].round(1).astype(str) + '%',
                    textposition='outside'
                ),
                row=row, col=col
            )
    
            # 100% line
            fig.add_hline(
                y=100,
                line_dash="dash",
                line_color="green",
                row=row, col=col
            )
    
            fig.update_xaxes(title_text="KPI", row=row, col=col)
            fig.update_yaxes(title_text="Achievement Rate (%)", row=row, col=col)
    
        def _add_trend_chart(self, fig: go.Figure, row: int, col: int):
            """Add time series trend (internal use)"""
            df = pd.DataFrame(self.data["control_chart"])
    
            # Only latest 7 days data
            if len(df) > 0:
                cutoff = datetime.now() - timedelta(days=7)
                df = df[df['timestamp'] >= cutoff]
    
            if len(df) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['value'],
                        mode='lines',
                        name='Trend',
                        line=dict(color='purple', width=2),
                        fill='tonexty'
                    ),
                    row=row, col=col
                )
    
                fig.update_xaxes(title_text="Date/Time", row=row, col=col)
                fig.update_yaxes(title_text="Measurement", row=row, col=col)
    
        def export_html(self, filename: str):
            """Export as HTML file
    
            Args:
                filename: Output filename
            """
            fig = self.create_dashboard()
            fig.write_html(filename)
            print(f"Dashboard exported to {filename}")
    
    
    # Usage example
    dashboard = QualityDashboard()
    
    # Add sample data
    np.random.seed(42)
    
    # Control chart data
    for i in range(20):
        timestamp = datetime.now() - timedelta(hours=20-i)
        value = np.random.normal(100, 2)
        dashboard.add_control_chart_data(
            timestamp=timestamp,
            value=value,
            target=100.0,
            ucl=106.0,
            lcl=94.0
        )
    
    # Defect data
    defect_types = ["Dimensional Defect", "Appearance Defect", "Functional Defect", "Packaging Defect", "Other"]
    defect_counts = [45, 28, 15, 8, 4]
    
    for defect_type, count in zip(defect_types, defect_counts):
        dashboard.add_defect_data(defect_type, count)
    
    # KPI data
    kpis = [
        ("Pass Rate", 98.5, 99.0, "%"),
        ("Delivery Compliance", 96.0, 95.0, "%"),
        ("Yield Rate", 92.0, 90.0, "%"),
        ("Complaints", 8, 10, "cases")
    ]
    
    for kpi_name, actual, target, unit in kpis:
        dashboard.add_kpi_data(kpi_name, actual, target, unit)
    
    # Display dashboard
    fig = dashboard.create_dashboard(title="Manufacturing Quality Dashboard - October 2025")
    # fig.show()  # Display in Jupyter, etc.
    
    # Save as HTML file
    dashboard.export_html("quality_dashboard.html")
    print("Interactive dashboard generated")

## 5.2 Automated Report Generation and Data Integration

### 5.2.1 Automated PDF/Excel Report Generation

A system that automatically generates reports from quality data and outputs them in PDF or Excel format.

Example 3: Automated Report Generation System
    
    
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from datetime import datetime
    import pandas as pd
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.chart import BarChart, Reference
    
    class AutoReportGenerator:
        """Automated Quality Report Generation System
    
        Automatically generates reports in PDF/Excel format based on quality data.
        Includes control charts, statistical summaries, and corrective action lists.
        """
    
        def __init__(self, report_period: str):
            self.report_period = report_period
            self.data = {
                "summary": {},
                "spc_data": pd.DataFrame(),
                "capa_data": [],
                "kpi_data": []
            }
    
        def add_summary_data(self, total_production: int, defect_count: int,
                            yield_rate: float, customer_complaints: int):
            """Add summary data
    
            Args:
                total_production: Total production
                defect_count: Defect count
                yield_rate: Yield rate
                customer_complaints: Customer complaints count
            """
            self.data["summary"] = {
                "total_production": total_production,
                "defect_count": defect_count,
                "yield_rate": yield_rate,
                "customer_complaints": customer_complaints,
                "defect_rate": (defect_count / total_production * 100)
                              if total_production > 0 else 0
            }
    
        def add_spc_data(self, df: pd.DataFrame):
            """Add SPC data
    
            Args:
                df: DataFrame of SPC measurement data
            """
            self.data["spc_data"] = df
    
        def add_capa_data(self, capa_list: list):
            """Add CAPA data
    
            Args:
                capa_list: List of CAPA information
            """
            self.data["capa_data"] = capa_list
    
        def add_kpi_data(self, kpi_list: list):
            """Add KPI data
    
            Args:
                kpi_list: List of KPI information
            """
            self.data["kpi_data"] = kpi_list
    
        def generate_excel_report(self, filename: str):
            """Generate Excel report
    
            Args:
                filename: Output filename
            """
            wb = openpyxl.Workbook()
    
            # Sheet 1: Summary
            ws_summary = wb.active
            ws_summary.title = "Summary"
    
            # Header
            ws_summary['A1'] = f"Quality Report - {self.report_period}"
            ws_summary['A1'].font = Font(size=16, bold=True)
            ws_summary['A1'].fill = PatternFill(start_color="4472C4", fill_type="solid")
            ws_summary['A1'].font = Font(size=16, bold=True, color="FFFFFF")
    
            # Summary data
            summary_data = [
                ["Item", "Value"],
                ["Total Production", self.data["summary"]["total_production"]],
                ["Defects", self.data["summary"]["defect_count"]],
                ["Defect Rate", f"{self.data['summary']['defect_rate']:.2f}%"],
                ["Yield Rate", f"{self.data['summary']['yield_rate']:.2f}%"],
                ["Customer Complaints", self.data["summary"]["customer_complaints"]]
            ]
    
            for row_idx, row_data in enumerate(summary_data, start=3):
                for col_idx, value in enumerate(row_data, start=1):
                    cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
                    if row_idx == 3:  # Header row
                        cell.font = Font(bold=True)
                        cell.fill = PatternFill(start_color="D9E1F2", fill_type="solid")
                    cell.alignment = Alignment(horizontal="left")
    
            # Sheet 2: SPC Data
            if not self.data["spc_data"].empty:
                ws_spc = wb.create_sheet(title="SPC Data")
    
                # Write DataFrame to sheet
                for r_idx, row in enumerate(
                    dataframe_to_rows(self.data["spc_data"], index=False, header=True),
                    start=1
                ):
                    for c_idx, value in enumerate(row, start=1):
                        cell = ws_spc.cell(row=r_idx, column=c_idx, value=value)
                        if r_idx == 1:  # Header
                            cell.font = Font(bold=True)
                            cell.fill = PatternFill(start_color="D9E1F2", fill_type="solid")
    
            # Sheet 3: CAPA
            if self.data["capa_data"]:
                ws_capa = wb.create_sheet(title="CAPA")
    
                capa_headers = ["CAPA ID", "Description", "Status", "Due Date"]
                for col_idx, header in enumerate(capa_headers, start=1):
                    cell = ws_capa.cell(row=1, column=col_idx, value=header)
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="D9E1F2", fill_type="solid")
    
                for row_idx, capa in enumerate(self.data["capa_data"], start=2):
                    ws_capa.cell(row=row_idx, column=1, value=capa["capa_id"])
                    ws_capa.cell(row=row_idx, column=2, value=capa["description"])
                    ws_capa.cell(row=row_idx, column=3, value=capa["status"])
                    ws_capa.cell(row=row_idx, column=4, value=capa["due_date"])
    
            # Sheet 4: KPI
            if self.data["kpi_data"]:
                ws_kpi = wb.create_sheet(title="KPI")
    
                kpi_headers = ["KPI Name", "Target", "Actual", "Achievement Rate"]
                for col_idx, header in enumerate(kpi_headers, start=1):
                    cell = ws_kpi.cell(row=1, column=col_idx, value=header)
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="D9E1F2", fill_type="solid")
    
                for row_idx, kpi in enumerate(self.data["kpi_data"], start=2):
                    ws_kpi.cell(row=row_idx, column=1, value=kpi["name"])
                    ws_kpi.cell(row=row_idx, column=2, value=kpi["target"])
                    ws_kpi.cell(row=row_idx, column=3, value=kpi["actual"])
                    ws_kpi.cell(row=row_idx, column=4,
                               value=f"{kpi['achievement']:.1f}%")
    
                    # Color code based on achievement rate
                    achievement = kpi['achievement']
                    cell = ws_kpi.cell(row=row_idx, column=4)
                    if achievement >= 100:
                        cell.fill = PatternFill(start_color="C6EFCE", fill_type="solid")
                    elif achievement >= 80:
                        cell.fill = PatternFill(start_color="FFEB9C", fill_type="solid")
                    else:
                        cell.fill = PatternFill(start_color="FFC7CE", fill_type="solid")
    
            wb.save(filename)
            print(f"Excel report exported to {filename}")
    
        def generate_pdf_report(self, filename: str):
            """Generate PDF report
    
            Args:
                filename: Output filename
            """
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
    
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor("#2c3e50"),
                spaceAfter=30
            )
    
            title = Paragraph(f"Quality Report  
    {self.report_period}", title_style)
            story.append(title)
            story.append(Spacer(1, 0.5*cm))
    
            # Summary section
            heading_style = styles['Heading2']
            story.append(Paragraph("1. Summary", heading_style))
            story.append(Spacer(1, 0.3*cm))
    
            summary_data = [
                ["Item", "Value"],
                ["Total Production", f"{self.data['summary']['total_production']:,}"],
                ["Defects", f"{self.data['summary']['defect_count']:,}"],
                ["Defect Rate", f"{self.data['summary']['defect_rate']:.2f}%"],
                ["Yield Rate", f"{self.data['summary']['yield_rate']:.2f}%"],
                ["Customer Complaints", f"{self.data['summary']['customer_complaints']} cases"]
            ]
    
            summary_table = Table(summary_data, colWidths=[8*cm, 6*cm])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
    
            story.append(summary_table)
            story.append(Spacer(1, 1*cm))
    
            # CAPA section
            if self.data["capa_data"]:
                story.append(Paragraph("2. Corrective and Preventive Actions (CAPA)", heading_style))
                story.append(Spacer(1, 0.3*cm))
    
                capa_data = [["CAPA ID", "Description", "Status", "Due Date"]]
                for capa in self.data["capa_data"]:
                    capa_data.append([
                        capa["capa_id"],
                        capa["description"][:40] + "..." if len(capa["description"]) > 40
                                                          else capa["description"],
                        capa["status"],
                        capa["due_date"]
                    ])
    
                capa_table = Table(capa_data, colWidths=[3*cm, 7*cm, 2.5*cm, 2.5*cm])
                capa_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
    
                story.append(capa_table)
                story.append(Spacer(1, 1*cm))
    
            # KPI section
            if self.data["kpi_data"]:
                story.append(Paragraph("3. KPI Achievement Status", heading_style))
                story.append(Spacer(1, 0.3*cm))
    
                kpi_data = [["KPI Name", "Target", "Actual", "Achievement Rate"]]
                for kpi in self.data["kpi_data"]:
                    kpi_data.append([
                        kpi["name"],
                        str(kpi["target"]),
                        str(kpi["actual"]),
                        f"{kpi['achievement']:.1f}%"
                    ])
    
                kpi_table = Table(kpi_data, colWidths=[5*cm, 3*cm, 3*cm, 3*cm])
                kpi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
    
                story.append(kpi_table)
    
            # Build PDF
            doc.build(story)
            print(f"PDF report exported to {filename}")
    
    
    # Helper function for openpyxl
    from openpyxl.utils.dataframe import dataframe_to_rows
    
    # Usage example
    report_gen = AutoReportGenerator(report_period="October 2025")
    
    # Add summary data
    report_gen.add_summary_data(
        total_production=50000,
        defect_count=750,
        yield_rate=98.5,
        customer_complaints=8
    )
    
    # Add CAPA data
    capa_list = [
        {
            "capa_id": "CAPA-20251001-001",
            "description": "Molding temperature exceeded control limits",
            "status": "In Progress",
            "due_date": "2025-10-31"
        },
        {
            "capa_id": "CAPA-20251005-002",
            "description": "Defect escaped due to inspection miss",
            "status": "Completed",
            "due_date": "2025-10-25"
        }
    ]
    report_gen.add_capa_data(capa_list)
    
    # Add KPI data
    kpi_list = [
        {"name": "Pass Rate", "target": 99.0, "actual": 98.5, "achievement": 99.5},
        {"name": "Delivery Compliance", "target": 95.0, "actual": 96.5, "achievement": 101.6},
        {"name": "Complaint Reduction", "target": 10, "actual": 8, "achievement": 125.0}
    ]
    report_gen.add_kpi_data(kpi_list)
    
    # Generate reports
    report_gen.generate_excel_report("quality_report_202510.xlsx")
    # report_gen.generate_pdf_report("quality_report_202510.pdf")  # Requires font configuration
    
    print("\nAutomated report generation complete")

### 5.2.2 Quality Data Management through Database Integration

Persist quality data using SQLite database and implement history management and query functionality.

Example 4: Database-Integrated Quality Management System
    
    
    import sqlite3
    import pandas as pd
    from datetime import datetime
    from typing import List, Dict, Optional
    
    class QualityDatabase:
        """Quality Database Management System
    
        Persist quality data using SQLite.
        Centrally manage measurement data, nonconformances, CAPAs, and audit results.
        """
    
        def __init__(self, db_file: str = "quality_data.db"):
            self.db_file = db_file
            self.conn = None
            self._initialize_database()
    
        def _initialize_database(self):
            """Initialize database and tables"""
            self.conn = sqlite3.connect(self.db_file)
            cursor = self.conn.cursor()
    
            # Measurement data table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                process_name TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                value REAL NOT NULL,
                target REAL,
                ucl REAL,
                lcl REAL,
                in_control BOOLEAN,
                operator TEXT,
                lot_number TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
    
            # Nonconformance table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS nonconformances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nc_id TEXT UNIQUE NOT NULL,
                date_detected DATE NOT NULL,
                product TEXT,
                defect_type TEXT NOT NULL,
                quantity INTEGER,
                severity TEXT,
                detected_by TEXT,
                status TEXT DEFAULT 'Open',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
    
            # CAPA table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS capas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                capa_id TEXT UNIQUE NOT NULL,
                nc_id TEXT,
                description TEXT NOT NULL,
                root_cause TEXT,
                corrective_action TEXT,
                preventive_action TEXT,
                assigned_to TEXT,
                due_date DATE,
                status TEXT DEFAULT 'Open',
                effectiveness_verified BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (nc_id) REFERENCES nonconformances(nc_id)
            )
            """)
    
            # Audit table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                audit_id TEXT UNIQUE NOT NULL,
                audit_date DATE NOT NULL,
                department TEXT NOT NULL,
                auditor TEXT NOT NULL,
                findings_count INTEGER DEFAULT 0,
                nc_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Planned',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
    
            self.conn.commit()
            print(f"Database initialization complete: {self.db_file}")
    
        def add_measurement(self, timestamp: datetime, process_name: str,
                           parameter_name: str, value: float,
                           target: float = None, ucl: float = None,
                           lcl: float = None, operator: str = None,
                           lot_number: str = None):
            """Record measurement data
    
            Args:
                timestamp: Measurement time
                process_name: Process name
                parameter_name: Parameter name
                value: Measurement value
                target: Target value
                ucl: Upper control limit
                lcl: Lower control limit
                operator: Operator
                lot_number: Lot number
            """
            cursor = self.conn.cursor()
    
            in_control = True
            if ucl is not None and lcl is not None:
                in_control = lcl <= value <= ucl
    
            cursor.execute("""
            INSERT INTO measurements
            (timestamp, process_name, parameter_name, value, target, ucl, lcl,
             in_control, operator, lot_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, process_name, parameter_name, value, target, ucl, lcl,
                  in_control, operator, lot_number))
    
            self.conn.commit()
    
        def add_nonconformance(self, nc_id: str, date_detected: datetime,
                              product: str, defect_type: str,
                              quantity: int, severity: str,
                              detected_by: str) -> int:
            """Record nonconformance
    
            Args:
                nc_id: Nonconformance ID
                date_detected: Detection date
                product: Product name
                defect_type: Defect type
                quantity: Quantity
                severity: Severity
                detected_by: Detected by
    
            Returns:
                ID of inserted record
            """
            cursor = self.conn.cursor()
    
            cursor.execute("""
            INSERT INTO nonconformances
            (nc_id, date_detected, product, defect_type, quantity, severity, detected_by)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (nc_id, date_detected, product, defect_type, quantity, severity, detected_by))
    
            self.conn.commit()
            return cursor.lastrowid
    
        def add_capa(self, capa_id: str, nc_id: str, description: str,
                    assigned_to: str, due_date: datetime) -> int:
            """Record CAPA
    
            Args:
                capa_id: CAPA ID
                nc_id: Related nonconformance ID
                description: Description
                assigned_to: Assigned to
                due_date: Due date
    
            Returns:
                ID of inserted record
            """
            cursor = self.conn.cursor()
    
            cursor.execute("""
            INSERT INTO capas
            (capa_id, nc_id, description, assigned_to, due_date)
            VALUES (?, ?, ?, ?, ?)
            """, (capa_id, nc_id, description, assigned_to, due_date))
    
            self.conn.commit()
            return cursor.lastrowid
    
        def update_capa(self, capa_id: str, root_cause: str = None,
                       corrective_action: str = None,
                       preventive_action: str = None,
                       status: str = None,
                       effectiveness_verified: bool = None):
            """Update CAPA
    
            Args:
                capa_id: CAPA ID
                root_cause: Root cause
                corrective_action: Corrective action
                preventive_action: Preventive action
                status: Status
                effectiveness_verified: Effectiveness verified
            """
            cursor = self.conn.cursor()
    
            updates = []
            params = []
    
            if root_cause is not None:
                updates.append("root_cause = ?")
                params.append(root_cause)
    
            if corrective_action is not None:
                updates.append("corrective_action = ?")
                params.append(corrective_action)
    
            if preventive_action is not None:
                updates.append("preventive_action = ?")
                params.append(preventive_action)
    
            if status is not None:
                updates.append("status = ?")
                params.append(status)
    
            if effectiveness_verified is not None:
                updates.append("effectiveness_verified = ?")
                params.append(effectiveness_verified)
    
            if updates:
                params.append(capa_id)
                sql = f"UPDATE capas SET {', '.join(updates)} WHERE capa_id = ?"
                cursor.execute(sql, params)
                self.conn.commit()
    
        def get_measurements(self, process_name: str = None,
                            start_date: datetime = None,
                            end_date: datetime = None) -> pd.DataFrame:
            """Get measurement data
    
            Args:
                process_name: Process name (for filtering)
                start_date: Start date
                end_date: End date
    
            Returns:
                DataFrame of measurement data
            """
            query = "SELECT * FROM measurements WHERE 1=1"
            params = []
    
            if process_name:
                query += " AND process_name = ?"
                params.append(process_name)
    
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
    
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
    
            query += " ORDER BY timestamp"
    
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
    
        def get_open_capas(self) -> pd.DataFrame:
            """Get open CAPAs
    
            Returns:
                DataFrame of open CAPAs
            """
            query = """
            SELECT c.capa_id, c.description, c.assigned_to, c.due_date,
                   c.status, n.defect_type, n.severity
            FROM capas c
            LEFT JOIN nonconformances n ON c.nc_id = n.nc_id
            WHERE c.status != 'Closed'
            ORDER BY c.due_date
            """
    
            df = pd.read_sql_query(query, self.conn)
            return df
    
        def get_defect_analysis(self, start_date: datetime = None,
                               end_date: datetime = None) -> pd.DataFrame:
            """Get defect analysis (Pareto chart data)
    
            Args:
                start_date: Start date
                end_date: End date
    
            Returns:
                DataFrame of defect type aggregation
            """
            query = """
            SELECT defect_type, SUM(quantity) as total_quantity, COUNT(*) as occurrence
            FROM nonconformances
            WHERE 1=1
            """
            params = []
    
            if start_date:
                query += " AND date_detected >= ?"
                params.append(start_date)
    
            if end_date:
                query += " AND date_detected <= ?"
                params.append(end_date)
    
            query += " GROUP BY defect_type ORDER BY total_quantity DESC"
    
            df = pd.read_sql_query(query, self.conn, params=params)
            return df
    
        def get_dashboard_metrics(self) -> Dict:
            """Get dashboard metrics
    
            Returns:
                Dictionary of key metrics
            """
            cursor = self.conn.cursor()
    
            # Current month data only
            start_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0)
    
            # Total measurements
            cursor.execute("""
            SELECT COUNT(*) FROM measurements WHERE timestamp >= ?
            """, (start_of_month,))
            total_measurements = cursor.fetchone()[0]
    
            # Out of control measurements
            cursor.execute("""
            SELECT COUNT(*) FROM measurements
            WHERE timestamp >= ? AND in_control = 0
            """, (start_of_month,))
            out_of_control = cursor.fetchone()[0]
    
            # Nonconformance count
            cursor.execute("""
            SELECT COUNT(*), COALESCE(SUM(quantity), 0)
            FROM nonconformances WHERE date_detected >= ?
            """, (start_of_month,))
            nc_count, nc_quantity = cursor.fetchone()
    
            # Open CAPAs
            cursor.execute("""
            SELECT COUNT(*) FROM capas WHERE status != 'Closed'
            """)
            open_capas = cursor.fetchone()[0]
    
            metrics = {
                "total_measurements": total_measurements,
                "out_of_control_rate": (out_of_control / total_measurements * 100)
                                       if total_measurements > 0 else 0,
                "nc_count": nc_count,
                "nc_quantity": nc_quantity,
                "open_capas": open_capas
            }
    
            return metrics
    
        def close(self):
            """Close database connection"""
            if self.conn:
                self.conn.close()
                print("Database connection closed")
    
    
    # Usage example
    db = QualityDatabase("quality_management.db")
    
    # Record measurement data
    for i in range(10):
        timestamp = datetime.now() - timedelta(hours=10-i)
        value = np.random.normal(100, 2)
    
        db.add_measurement(
            timestamp=timestamp,
            process_name="Molding Temperature",
            parameter_name="Temperature",
            value=value,
            target=100.0,
            ucl=106.0,
            lcl=94.0,
            operator="Operator A",
            lot_number=f"LOT-2025{i:03d}"
        )
    
    # Record nonconformance
    nc_id = "NC-20251026-001"
    db.add_nonconformance(
        nc_id=nc_id,
        date_detected=datetime.now(),
        product="Product X",
        defect_type="Dimensional Defect",
        quantity=50,
        severity="Major",
        detected_by="Inspector B"
    )
    
    # Record CAPA
    capa_id = "CAPA-20251026-001"
    db.add_capa(
        capa_id=capa_id,
        nc_id=nc_id,
        description="Dimensional defect due to mold wear",
        assigned_to="Manufacturing Manager",
        due_date=datetime.now() + timedelta(days=30)
    )
    
    # Update CAPA
    db.update_capa(
        capa_id=capa_id,
        root_cause="Lack of mold maintenance",
        corrective_action="Replace mold and perform measurements",
        preventive_action="Establish regular maintenance schedule",
        status="In Progress"
    )
    
    # Get data
    print("\n=== Measurement Data ===")
    measurements_df = db.get_measurements(process_name="Molding Temperature")
    print(measurements_df.head())
    
    print("\n=== Open CAPAs ===")
    open_capas_df = db.get_open_capas()
    print(open_capas_df)
    
    print("\n=== Dashboard Metrics ===")
    metrics = db.get_dashboard_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Close database connection
    db.close()

## 5.3 Quality Prediction with Machine Learning

### 5.3.1 Defect Prediction System (Machine Learning Model)

Build a machine learning model that predicts defects from manufacturing parameters.

Example 5: Machine Learning-Based Defect Prediction System
    
    
    # Due to space limitations, this example has been abbreviated
    # Please refer to the full implementation for complete defect prediction code
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import joblib
    from typing import Dict, Tuple
    
    class DefectPredictionSystem:
        """Machine Learning-Based Defect Prediction System
    
        Predicts defect occurrence from manufacturing process parameters.
        Achieves high prediction accuracy with Random Forest model.
        """
    
        def __init__(self):
            self.model = None
            self.scaler = None
            self.feature_names = None
            self.is_trained = False
    
        def train(self, X_train, y_train, n_estimators: int = 100, max_depth: int = 10):
            """Train model"""
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_train, y_train)
            self.is_trained = True
    
        def predict(self, X: pd.DataFrame) -> Dict:
            """Predict defects"""
            if not self.is_trained:
                raise ValueError("Model not trained")
    
            X_scaled = self.scaler.transform(X[self.feature_names])
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0, 1]
    
            # Determine risk level
            if probability >= 0.7:
                risk_level = "High"
            elif probability >= 0.4:
                risk_level = "Medium"
            else:
                risk_level = "Low"
    
            return {
                "prediction": "Defect" if prediction == 1 else "Pass",
                "defect_probability": probability,
                "risk_level": risk_level
            }

## Summary

In this chapter, we learned practical implementation methods for quality control automation using Python.

#### 📚 Key Points

  * **Real-time Monitoring** : Automatic abnormal pattern detection using Western Electric Rules
  * **Interactive Dashboard** : Visualization with Plotly and HTML output
  * **Automated Report Generation** : Periodic report creation in PDF/Excel format
  * **Database Integration** : Persistence and history management using SQLite
  * **Machine Learning Prediction** : Defect prediction model using Random Forest
  * **API Integration** : Data collection automation through MES/ERP system integration
  * **Sampling Plan** : Automatic generation of optimal plans based on statistical methods
  * **End-to-End Integration** : QMS workflow integrating all components

#### 💡 Practical Application

By combining these systems, you can achieve advanced quality management such as:

  * 24-hour automated monitoring and alerts for manufacturing processes
  * Real-time dashboard for executive management
  * Defect prediction and preventive maintenance using AI
  * Fully automated quality reporting
  * Data-driven continuous improvement activities

[← Chapter 4: Quality Standards and ISO 9001](<chapter-4.html>) [Return to Series Index →](<index.html>)

## References

  1. Montgomery, D. C. (2019). _Design and Analysis of Experiments_ (9th ed.). Wiley.
  2. Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). _Statistics for Experimenters: Design, Innovation, and Discovery_ (2nd ed.). Wiley.
  3. Seborg, D. E., Edgar, T. F., Mellichamp, D. A., & Doyle III, F. J. (2016). _Process Dynamics and Control_ (4th ed.). Wiley.
  4. McKay, M. D., Beckman, R. J., & Conover, W. J. (2000). "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code." _Technometrics_ , 42(1), 55-61.

### Disclaimer

  * This content is provided solely for educational, research, and informational purposes and does not constitute professional advice (legal, accounting, technical warranty, etc.).
  * This content and accompanying code examples are provided "AS IS" without any warranty, express or implied, including but not limited to merchantability, fitness for a particular purpose, non-infringement, accuracy, completeness, operation, or safety.
  * The author and Tohoku University assume no responsibility for the content, availability, or safety of external links, third-party data, tools, libraries, etc.
  * To the maximum extent permitted by applicable law, the author and Tohoku University shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use, execution, or interpretation of this content.
  * The content may be changed, updated, or discontinued without notice.
  * The copyright and license of this content are subject to the stated conditions (e.g., CC BY 4.0). Such licenses typically include no-warranty clauses.
