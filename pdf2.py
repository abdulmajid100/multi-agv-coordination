from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Create a new PDF
output_pdf = "output_high_res.pdf"
c = canvas.Canvas(output_pdf, pagesize=letter)

# Add text with high resolution
text = """
Institution of MECHANICAL ENGINEERS
Original Article
A digital twin-driven dynamic path planning approach for multiple automatic guided vehicles based on deep reinforcement learning
"""

# Set font and size
c.setFont("Helvetica", 12)

# Write text to the PDF
x, y = 50, 750  # Starting position
for line in text.split("\n"):
    c.drawString(x, y, line)
    y -= 15  # Move to the next line

# Save the PDF
c.save()

print(f"High-resolution text PDF saved as {output_pdf}")