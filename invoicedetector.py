import os
import cv2
import pytesseract
import easyocr
import numpy as np
import re
import json
import time
import torch
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import io
import csv
import logging
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("invoice_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
OUTPUT_DIR = "processed_data"
DEBUG_DIR = "debug_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
data_list = []
SAMPLE_IMAGE_PATH = os.path.join('static', 'sample_invoice.jpg')

# OCR Configuration
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    logger.info("Tesseract path configured")
except Exception as e:
    logger.warning(f"Tesseract configuration error: {str(e)}")

# Initialize EasyOCR
try:
    reader = easyocr.Reader(
        ['en'],
        gpu=torch.cuda.is_available(),
        model_storage_directory='model_cache',
        download_enabled=True,
        quantize=True
    )
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"EasyOCR initialization failed: {str(e)}")
    reader = None

def save_debug_image(image, prefix="preprocess"):
    """Save intermediate images for debugging"""
    try:
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(DEBUG_DIR, filename)
        cv2.imwrite(path, image)
        logger.debug(f"Saved debug image: {path}")
        return path
    except Exception as e:
        logger.error(f"Error saving debug image: {str(e)}")
        return None

def preprocess_image(image):
    """Advanced image preprocessing pipeline"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        save_debug_image(gray, "gray")
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
        save_debug_image(denoised, "denoised")
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        save_debug_image(enhanced, "enhanced")
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            enhanced, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            51, 
            11
        )
        save_debug_image(thresh, "threshold")
        
        # Morphological operations to clean up text
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        save_debug_image(morph, "morph")
        
        return morph
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return gray

def extract_text_with_layout(image_data):
    """Robust OCR with improved error handling"""
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Preprocess
        processed = preprocess_image(img)
        
        # Try Tesseract with layout analysis
        try:
            logger.info("Running Tesseract with layout analysis")
            custom_config = r'--oem 1 --psm 11 -c preserve_interword_spaces=1'
            
            # Get OCR data as dictionary
            ocr_data = pytesseract.image_to_data(
                processed, 
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Check if we got valid results
            if not ocr_data.get('text') or len(ocr_data['text']) == 0:
                raise Exception("Tesseract returned empty text")
                
            # Build structured text
            structured_text = []
            ocr_lines = []
            
            # Group text by line
            current_line = ""
            last_y = -1
            
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    if last_y != -1 and abs(ocr_data['top'][i] - last_y) > 10:
                        structured_text.append({
                            'text': current_line.strip(),
                            'x': ocr_data['left'][i],
                            'y': last_y
                        })
                        ocr_lines.append(current_line.strip())
                        current_line = ""
                    
                    current_line += text + " "
                    last_y = ocr_data['top'][i]
            
            # Add last line
            if current_line.strip():
                structured_text.append({
                    'text': current_line.strip(),
                    'x': ocr_data['left'][i],
                    'y': last_y
                })
                ocr_lines.append(current_line.strip())
                
            return "\n".join(ocr_lines), structured_text
            
        except Exception as tess_err:
            logger.warning(f"Tesseract layout failed: {str(tess_err)}")
            # Fallback to simple OCR
            ocr_text = pytesseract.image_to_string(
                processed,
                config='--oem 1 --psm 6'
            )
            return ocr_text, []
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        # Final fallback to EasyOCR
        if reader:
            try:
                logger.info("Falling back to EasyOCR")
                results = reader.readtext(img, paragraph=True, detail=0)
                ocr_text = "\n".join(results)
                return ocr_text, []
            except:
                return "OCR failed: " + str(e), []
        return "OCR error: " + str(e), []

def perform_ocr(image_data):
    """Perform OCR using best available method with fallback"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_data)
        
        # Try Tesseract first
        try:
            logger.info("Trying Tesseract OCR")
            custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            ocr_text = pytesseract.image_to_string(processed_img, config=custom_config)
            logger.debug(f"Tesseract output:\n{ocr_text}")
            
            if len(ocr_text.strip()) < 50:  # If too little text
                raise Exception("Tesseract returned insufficient text")
                
            return ocr_text.strip()
        except Exception as tess_err:
            logger.warning(f"Tesseract failed: {str(tess_err)}")
            
            # Fallback to EasyOCR
            if reader:
                logger.info("Falling back to EasyOCR")
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Resize for EasyOCR
                height, width = img.shape[:2]
                max_dim = max(width, height)
                scale = 1600 / max_dim if max_dim > 1600 else 1.0
                if scale != 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                
                results = reader.readtext(img, paragraph=True, detail=0)
                ocr_text = "\n".join(results)
                logger.debug(f"EasyOCR output:\n{ocr_text}")
                return ocr_text.strip()
            
            # Last resort: return Tesseract result anyway
            return ocr_text.strip() if ocr_text.strip() else "OCR failed: No text detected"
            
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        return "OCR error: " + str(e)

def extract_invoice_data(ocr_text, structured_data):
    """Improved invoice data extraction with fallbacks"""
    logger.info("Extracting invoice data")
    
    # Initialize result structure
    result = {
        "invoice_number": "N/A",
        "date": "N/A",
        "seller": {
            "name": "Unknown",
            "address": "Unknown",
            "tax_id": "Unknown",
            "iban": "Unknown"
        },
        "totals": {
            "net": "0.00",
            "vat": "0.00",
            "gross": "0.00"
        },
        "items": []
    }
    
    # Helper function to find value after label
    def find_field(pattern, text):
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    # Extract basic fields with multiple patterns
    try:
        # Invoice number patterns
        patterns = [
            r"invoice\s*#?\s*[:]?\s*(\w[\w-]*)",
            r"invoice\s*no\.?\s*[:]?\s*(\w[\w-]*)",
            r"bill\s*#?\s*[:]?\s*(\w[\w-]*)"
        ]
        for pattern in patterns:
            inv_num = find_field(pattern, ocr_text)
            if inv_num:
                result["invoice_number"] = inv_num
                break
        
        # Date patterns
        date_match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", ocr_text)
        if date_match:
            result["date"] = date_match.group(1)
        
        # Seller name patterns
        seller_patterns = [
            r"from:\s*(.+)",
            r"seller:\s*(.+)",
            r"vendor:\s*(.+)",
            r"supplier:\s*(.+)"
        ]
        for pattern in seller_patterns:
            seller = find_field(pattern, ocr_text)
            if seller:
                result["seller"]["name"] = seller
                break
        
        # Tax ID patterns
        tax_patterns = [
            r"tax\s*id\s*[:]?\s*(\w+)",
            r"vat\s*id\s*[:]?\s*(\w+)",
            r"tax\s*#\s*[:]?\s*(\w+)"
        ]
        for pattern in tax_patterns:
            tax_id = find_field(pattern, ocr_text)
            if tax_id:
                result["seller"]["tax_id"] = tax_id
                break
        
        # IBAN patterns
        iban_match = re.search(r"IBAN[:]?\s*([A-Z0-9]{15,34})", ocr_text, re.IGNORECASE)
        if iban_match:
            result["seller"]["iban"] = iban_match.group(1)
        
        # Total amounts
        total_patterns = [
            r"total\s*amount\s*[:]?\s*([\d,]+\.\d{2})",
            r"grand\s*total\s*[:]?\s*([\d,]+\.\d{2})",
            r"amount\s*due\s*[:]?\s*([\d,]+\.\d{2})",
            r"balance\s*due\s*[:]?\s*([\d,]+\.\d{2})"
        ]
        for pattern in total_patterns:
            total = find_field(pattern, ocr_text)
            if total:
                result["totals"]["gross"] = total.replace(",", "")
                break
        
        # Try to find items table using position data if available
        if structured_data:
            # Sort by Y position then X position
            structured_data.sort(key=lambda x: (x.get('y', 0), x.get('x', 0)))
            
            # Find table header positions
            header_keywords = ["item", "description", "qty", "price", "amount"]
            header_lines = []
            
            for i, item in enumerate(structured_data):
                if any(kw in item['text'].lower() for kw in header_keywords):
                    header_lines.append(item)
            
            if header_lines:
                # Find the main header (lowest Y position)
                main_header = min(header_lines, key=lambda x: x['y'])
                
                # Extract items below the header
                items = []
                for item in structured_data:
                    if item['y'] > main_header['y'] + 10:  # Below header
                        items.append(item['text'])
                
                # Simple item parsing
                for i, item_text in enumerate(items[:10]):  # Limit to 10 items
                    parts = re.split(r'\s{2,}', item_text)  # Split on multiple spaces
                    if len(parts) > 2:
                        result["items"].append({
                            "item_number": i+1,
                            "description": " ".join(parts[:-2]),
                            "quantity": parts[-2],
                            "unit_price": parts[-1].replace(",", ""),
                            "total_price": ""
                        })
        
        # Fallback: simple regex item extraction
        if not result["items"]:
            item_matches = re.finditer(r"(\d+)\s+(.+?)\s+(\d+)\s+([\d.,]+)\s+([\d.,]+)", ocr_text)
            for i, match in enumerate(item_matches, 1):
                result["items"].append({
                    "item_number": i,
                    "description": match.group(2).strip(),
                    "quantity": match.group(3),
                    "unit_price": match.group(4).replace(",", ""),
                    "total_price": match.group(5).replace(",", "")
                })
        
        logger.info(f"Extracted invoice data: {json.dumps(result, indent=2)}")
        return result
        
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        return result
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    file = request.files['image']
    try:
        logger.info(f"Processing started for {file.filename}")
        start_time = time.time()
        
        # Read image data
        image_data = file.read()
        file_size = len(image_data)
        logger.info(f"Received image: {file.filename} ({file_size/1024:.1f} KB)")
        
        # Perform OCR with layout analysis
        ocr_text, structured_data = extract_text_with_layout(image_data)
        logger.info(f"OCR completed for {file.filename}")
        logger.debug(f"OCR Text:\n{ocr_text}")
        
        # Extract structured data
        parsed_data = extract_invoice_data(ocr_text, structured_data)
        
        # Store results
        data_list.append({
            "filename": file.filename,
            "ocr_text": ocr_text,
            "parsed_data": parsed_data
        })
        
        # Save to JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file.filename)[0]}.json")
        with open(output_path, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        
        proc_time = time.time() - start_time
        logger.info(f"Processed {file.filename} in {proc_time:.2f}s")
        return jsonify({
            "success": True,
            "filename": file.filename,
            "data": parsed_data,
            "ocr_text": ocr_text,
            "processing_time": f"{proc_time:.2f}s"
        })
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "filename": file.filename
        }), 500
    finally:
        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@app.route('/demo', methods=['GET'])
def demo_processing():
    try:
        logger.info("Processing sample invoice")
        start_time = time.time()
        
        if not os.path.exists(SAMPLE_IMAGE_PATH):
            # Create sample image if not exists
            sample_img = np.zeros((500, 800, 3), dtype=np.uint8)
            cv2.putText(sample_img, "Sample Invoice", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(sample_img, "Invoice No: INV-2023-001", (50, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
            cv2.putText(sample_img, "Date: 2023-10-15", (50, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
            cv2.putText(sample_img, "Total: $450.00", (50, 350), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)
            cv2.imwrite(SAMPLE_IMAGE_PATH, sample_img)
            logger.info(f"Created sample invoice at {SAMPLE_IMAGE_PATH}")
        
        # Read and process sample image
        with open(SAMPLE_IMAGE_PATH, 'rb') as f:
            image_data = f.read()
        
        ocr_text = perform_ocr(image_data)
        parsed_data = extract_invoice_data(ocr_text)
        
        data_list.append({
            "filename": "sample_invoice.jpg",
            "ocr_text": ocr_text,
            "parsed_data": parsed_data
        })
        
        proc_time = time.time() - start_time
        logger.info(f"Processed sample invoice in {proc_time:.2f}s")
        return jsonify({
            "success": True,
            "filename": "sample_invoice.jpg",
            "data": parsed_data,
            "ocr_text": ocr_text,
            "processing_time": f"{proc_time:.2f}s"
        })
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Demo processing failed: {str(e)}"
        }), 500

@app.route('/export', methods=['GET'])
def export_data():
    if not data_list:
        return jsonify({"error": "No data to export"}), 404
    
    try:
        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = [
            "Filename", "Invoice Number", "Date", 
            "Seller Name", "Tax ID", "IBAN",
            "Net Total", "VAT", "Gross Total",
            "Item Count"
        ]
        writer.writerow(headers)
        
        # Write data rows
        for entry in data_list:
            data = entry["parsed_data"]
            seller = data.get("seller", {})
            totals = data.get("totals", {})
            
            writer.writerow([
                entry["filename"],
                data.get("invoice_number", "N/A"),
                data.get("date", "N/A"),
                seller.get("name", "Unknown"),
                seller.get("tax_id", "Unknown"),
                seller.get("iban", "Unknown"),
                totals.get("net", "0.00"),
                totals.get("vat", "0.00"),
                totals.get("gross", "0.00"),
                len(data.get("items", []))
            ])
        
        # Create file response
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name='invoice_export.csv'
        )
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return jsonify({"error": "CSV generation failed"}), 500

@app.route('/debug_images/<path:filename>')
def debug_image(filename):
    return send_from_directory(DEBUG_DIR, filename)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create directories if not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Create sample image if needed
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        sample_img = np.zeros((500, 800, 3), dtype=np.uint8)
        cv2.putText(sample_img, "Sample Invoice", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(sample_img, "Invoice No: INV-2023-001", (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        cv2.putText(sample_img, "Date: 2023-10-15", (50, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
        cv2.putText(sample_img, "Total: $450.00", (50, 350), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)
        cv2.imwrite(SAMPLE_IMAGE_PATH, sample_img)
        logger.info(f"Created sample invoice at {SAMPLE_IMAGE_PATH}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)