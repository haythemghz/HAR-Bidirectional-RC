import zipfile
import xml.etree.ElementTree as ET
import sys

def get_docx_text(path):
    try:
        with zipfile.ZipFile(path) as document:
            xml_content = document.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            # Namespaces usually required for findall with prefixes, but we can ignore or accept all
            # Simple extraction:
            text = []
            for elem in tree.iter():
                if elem.tag.endswith('}t'):
                    if elem.text:
                        text.append(elem.text)
                elif elem.tag.endswith('}p'):
                    text.append('\n')
            return ''.join(text)
    except Exception as e:
        return f"Error reading docx: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(get_docx_text(sys.argv[1]))
    else:
        print("Usage: python read_docx.py <path_to_docx>")
