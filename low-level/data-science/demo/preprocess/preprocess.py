import pandas as pd
import csv

def read_csv():
    csv = pd.read_csv('data/original.csv', delimiter=';')
    headers = list(csv.columns.values)
    return {
        "items": csv.to_dict('records'),
        "headers": headers
    }

def write_csv(items, headers):
    with open("data/processed.csv", "w", newline="") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(headers)
        for item in items:
            writer.writerow(item.values())

def process(items):
    items = clear_descriptions(items)
    items = remove_items(items)
    items = sorted(items, key=lambda k: k['Label'])
    return items

def clear_descriptions(items):
    for item in items:
        description = item["Question"]
        item["Question"] = description.split('Best regards', 1)[0]
    return items

def remove_items(items):
    new_items = []
    for item in items:
        description = item["Question"]
        if len(description) < 10 or len(description) > 100:
            continue
        new_items.append(item)
    return new_items

csv_data = read_csv()
items = process(csv_data["items"])
write_csv(items, csv_data["headers"])
