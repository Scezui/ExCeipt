def create_csv_from_json_file(json_file, output_file):
    # Loading JSON data from file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Creating a dictionary to store labels and corresponding texts
    label_texts = {}
    for item in data:
        for output_item in item['output']:
            label = output_item['label']
            text = output_item['text']
            
            if label not in label_texts:
                label_texts[label] = []
            label_texts[label].append(text)

    # Order of columns as requested
    column_order = [
        'RECEIPTNUMBER', 'MERCHANTNAME', 'MERCHANTADDRESS', 
        'TRANSACTIONDATE', 'TRANSACTIONTIME', 'ITEMS', 
        'PRICE', 'TOTAL', 'VATTAX'
    ]

    # Writing data to CSV file with ordered columns
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=column_order)
        csv_writer.writeheader()
        
        # Determining the maximum number of texts for any label
        max_texts = max(len(label_texts[label]) if label in label_texts else 0 for label in column_order)
        
        # Filling in the CSV rows with texts for each label
        for i in range(max_texts):
            row_data = {
                label: label_texts[label][i] if label in label_texts and i < len(label_texts[label]) else '' 
                for label in column_order
            }
            csv_writer.writerow(row_data)