import fitz  # PyMuPDF

# Open the original PDF
input_pdf = "C:/Users/majmo/Downloads/filee100 (1).pdf"
output_pdf = "C:/Users/majmo/Downloads/output_high_res.pdf"

# Create a new PDF
doc = fitz.open(input_pdf)
new_doc = fitz.open()

# Iterate through pages and recreate them
for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()  # Extract text from the page

    # Create a new page in the new document
    new_page = new_doc.new_page(width=page.rect.width, height=page.rect.height)

    # Add the text to the new page
    new_page.insert_text((50, 50), text, fontsize=12, fontname="helv", color=(0, 0, 0))

# Save the new PDF
new_doc.save(output_pdf)
new_doc.close()
doc.close()

print(f"High-resolution text PDF saved as {output_pdf}")