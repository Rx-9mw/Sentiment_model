import csv

def convert_label_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        
        for line in infile:
            line = line.strip()
            if not line.startswith("__label__"):
                continue
            
            # Extract label and content
            label_end = line.find(" ")
            label = line[len("__label__"):label_end]
            content = line[label_end + 1:]
            
            # Split on the first colon
            if ":" in content:
                title, review = content.split(":", 1)
                title = title.strip()
                review = review.strip()
            else:
                # Fallback in case no colon
                title = ""
                review = content.strip()
            
            writer.writerow([label, title, review])

    print(f"✅ Converted lines written to: {output_file}")

# Example usage
convert_label_format("./Data/train.ft.txt", "converted_reviews.csv")