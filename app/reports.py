from reportlab.pdfgen import canvas

def export_pdf(risk, user):
    filename = f"{user}_report.pdf"
    c = canvas.Canvas(filename)
    c.drawString(100,750,"Customer Churn Risk Report")
    c.drawString(100,720,f"User: {user}")
    c.drawString(100,690,f"Risk Score: {risk:.2f}")
    c.save()
    return filename
