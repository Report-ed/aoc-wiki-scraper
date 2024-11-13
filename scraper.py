import json
import os
import re
import cv2
import http.client
from io import BytesIO, StringIO
import gzip
import zlib
import glob
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from resources.reserved_words import reserved_words

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_WIKI_URL = 'https://ashesofcreation.wiki'
HEADERS = {'User-Agent': 'Build_Planner/0.1.1 (Build_Planner/; discord:.reported)'}

def _create_connection(parsed_url):
    return http.client.HTTPSConnection(parsed_url.netloc)

def fetch_html_content(url):
    """Fetch HTML content from the given URL."""
    parsed_url = urlparse(url)
    connection = _create_connection(parsed_url)
    try:
        connection.request("GET", parsed_url.path, headers=HEADERS)
        response = connection.getresponse()
        if response.status == 200:
            return response.read()
        logger.error(f"HTTP error occurred: {response.status} {response.reason}")
    except Exception as e:
        logger.error(f"Exception during HTTP request: {e}")
    finally:
        connection.close()
    return None

def extract_description(soup):
    """Extract item description."""
    description_tag = soup.select_one('dl dd i')
    return description_tag.get_text(strip=True) if description_tag else None

def extract_general_details(soup):
    """Extract general details from the wikitable."""
    details_table = soup.select_one('table.wikitable')
    if details_table:
        details = {}
        for row in details_table.select('tr'):
            header = row.find('th').get_text(strip=True)
            data = row.find('td').get_text(strip=True)
            details[header] = data
        return details
    return None

def extract_drops(soup):
    """Extract drops from section."""
    return [li.get_text(strip=True) for li in soup.select('#Drops_from + ul li')]

def extract_stat_tables(soup):
    """Extract stat tables data with rarity caption."""
    stat_tables = {}
    for h3 in soup.find_all('h3'):
        table = h3.find_next_sibling('table', class_='wikitable stat-table')
        if table:
            rarity = h3.get_text(strip=True)
            table_data = []
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            for row in table.find_all('tr')[1:]:  # Skip header row
                values = [td.get_text(strip=True) for td in row.find_all('td')]
                if values:
                    table_data.append(dict(zip(headers, values)))
            stat_tables[rarity] = table_data
    return json.dumps(stat_tables)  # Serialize the dictionary to a JSON string

def extract_recipe_tables(soup):
    """Extract recipe table data with captions into JSON format."""
    recipe_tables = {}
    # Locate recipe tables
    for table in soup.select('table.wikitable.recipe-table'):
        caption_tag = table.find('caption')
        caption = (caption_tag.get_text(strip=True).replace(" ", "") 
                   if caption_tag else "UnknownCaption")
        
        # Log: Table caption
        logger.info(f"Processing table with caption: {caption}")
        # Initialize sub-lists
        required_components = []
        selectable_components = []
        current_section = None
        # Process rows
        for row in table.find_all('tr'):
            th_cells = row.find_all('th')
            # Detects section headers like "Required components" or "Selectable components"
            if th_cells and len(th_cells) == 1:  
                header_text = th_cells[0].get_text(strip=True)
                if "Required components" in header_text:
                    current_section = required_components
                elif "Selectable components" in header_text:
                    current_section = selectable_components
            # Handle component rows only if current_section is valid
            if current_section is not None:
                td_cells = row.find_all('td')
                if len(td_cells) >= 2:
                    component_name = row.find('th').get_text(strip=True)
                    rarity = td_cells[0].get_text(strip=True) if td_cells[0].get_text(strip=True) else "N/A"
                    quantity = td_cells[1].get_text(strip=True)
                    
                    logger.debug(f"Extracted Component - Name: {component_name}, Rarity: {rarity}, Quantity: {quantity}")
                    current_section.append({
                        "Component": component_name,
                        "Quantity": int(quantity),
                        "Rarity": rarity
                    })
        
        # Log if no components were appended
        if not required_components and not selectable_components:
            logger.warning(f"No components were found for caption: {caption}")
        # Store results
        if required_components or selectable_components:
            recipe_tables[caption] = {
                "Required Components": required_components,
                "Selectable Components": selectable_components
            }
    logger.debug(f"Completed parsing table, results: {recipe_tables}")
    return json.dumps(recipe_tables)

def fetch_item_info(item_name, wiki_link, page_name):
    url = f"{BASE_WIKI_URL}{wiki_link}"
    html_content = fetch_html_content(url)
    if not html_content:
        logger.error(f"Failed to fetch content for {item_name}")
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    description = extract_description(soup)
    details = extract_general_details(soup)
    drops_from = json.dumps(extract_drops(soup))  # Convert list to JSON string
    stat_tables = extract_stat_tables(soup)
    recipe_tables = extract_recipe_tables(soup)
    
    # Include page_name in the returned dictionary
    return {
        'Item': item_name,
        'Description': description,
        'Details': details,
        'Drops_From': drops_from,
        'Stat_Tables': stat_tables,
        'Recipe_Table': recipe_tables,
        'Page_Name': page_name,
    }

def create_db_tables(df, database, table_name):
    """Create and store data in MySQL database."""
    try:
        table_name = table_name.lower().replace(" ", "_")
        # Convert dictionary columns to JSON strings
        for column in df.columns:
            if df[column].apply(lambda x: isinstance(x, dict)).any():
                df[column] = df[column].apply(json.dumps)
        engine = create_engine(f'mysql+mysqlconnector://xxxxx:xxxxxx@aws.connect.psdb.cloud/{database}')
        with engine.connect() as connection:
            df.to_sql(table_name, con=connection, if_exists='append', index=False, method='multi')
            logger.info(f"Data appended to table {table_name} in database {database} successfully.")
    except SQLAlchemyError as error:
        logger.error(f'Failed to store data in MySQL: {error}')

def truncate_table(database, table_name):
    """Truncate the table in the MySQL database."""
    try:
        table_name = table_name.lower().replace(" ", "_")
        engine = create_engine(f'mysql+mysqlconnector://root:root@localhost/{database}')
        with engine.connect() as connection:
            connection.execute(text(f"TRUNCATE TABLE {table_name}"))
            logger.info(f"Table {table_name} truncated successfully.")
    except SQLAlchemyError as error:
        logger.error(f'Failed to truncate table in MySQL: {error}')

def decompress_response(response):
    encoding = response.getheader('Content-Encoding')
    if (encoding == 'gzip'):
        buf = BytesIO(response.read())
        with gzip.GzipFile(fileobj=buf) as f:
            return f.read()
    elif (encoding == 'deflate'):
        return zlib.decompress(response.read())
    return response.read()

def request_json(category, page):
    
    category_path = f'static/{category}/{page}/'.replace('Template:List_of_', '')
    url = f'{BASE_WIKI_URL}/api.php?action=parse&format=json&page={page}&prop=text&wrapoutputclass=mw-parser-output'
    parsed_url = urlparse(url)
    connection = http.client.HTTPSConnection(parsed_url.netloc)
    headers = {
        'User-Agent':
        'Build_Planner/0.1.1 (Build_Planner/; discord:.reported)',
        'Content-Type': 'application/json; charset=utf-8',
        'Referer': BASE_WIKI_URL
    }
    try:
        connection.request("GET",
                           parsed_url.path + "?" + parsed_url.query,
                           headers=headers)
        response = connection.getresponse()
        if (response.status == 200):
            data = decompress_response(response).decode('utf-8')
            json_data = json.loads(data)
            page = page.replace('Template:List_of_','')
            os.makedirs(category_path, exist_ok=True)
            with open(f'{category_path}/{page}.json', 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
            return True
        print(f'HTTP error occurred: {response.status} {response.reason}')
    except Exception as err:
        print(f'An error occurred: {err}')
    finally:
        connection.close()
        pretty_and_save(f'{category_path}/{page}.json',
                        f'{category_path}/{page}.html')
        with open(f'static/{category}/{page}/{page}.html', 'r', encoding='utf-8') as f:
            generate_csvs(f.read(), category_path, category, page)
    return False

def generate_csvs(html_content, category_path, category, page):
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table',
                           class_=[
                               'wikitable sortable',
                               'wikitable skilltable sortable',
                               'wikitable skilltable sortable skill'
                           ])
    for table in tables:
        heading = table.find_previous(lambda tag: tag.name.startswith('h') and tag.find('span', id=True) is not None)
        heading_id = heading.find('span', id=True)['id'] if heading else page.replace('Template:', '')
        heading_id = heading_id.replace('Skills/', 'Skills_')
        if heading_id == 'List_of_flasks':
            heading_id = 'List_of_potions'
        table_html = StringIO(str(table))
        df = pd.read_html(table_html, flavor='bs4')[0]
        icon_dict, href_dict = extract_icons_and_hrefs(table)
        df['Icon'] = df.iloc[:, 0].map(icon_dict)
        df['Wiki_Link'] = df.iloc[:, 0].map(href_dict)
        df['ImagePath'] = None
        img_folder = os.path.join(
            f'static/{category}/{page.replace("Template:List_of_", "")}/imgs/')
        os.makedirs(category_path, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)
        for row in df.itertuples():
            image_path = row.Icon
            image_name = os.path.join(img_folder, f'{row[1]}.png')
            if pd.notna(image_path):
                img_path = download_image(image_path, image_name)
                df.at[row.Index, 'ImagePath'] = img_path
        # Rename columns to avoid SQL reserved words
        df.columns = [sanitize_column_name(name) for name in df.columns]
        df['itemType'] = page
        df.to_csv(os.path.join(category_path, f'{heading_id}.csv'), index=False)

def _get_safe_filename(file_name):
    # Remove any characters that are not allowed in a file name.
    return re.sub(r'[<>:"\\|?*]', '', file_name)

def convert_to_webp(png_path):
    webp_path = png_path.replace('.png', '.webp')
    img = cv2.imread(png_path)
    if img is not None:
        cv2.imwrite(webp_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), 75])
    os.remove(png_path)
    return webp_path

def _download_and_convert_image(image_path, save_path):
    # Use the sanitized name for the file
    save_path = _get_safe_filename(save_path)
    parsed_url = urlparse(f"{BASE_WIKI_URL}{image_path}")
    connection = http.client.HTTPSConnection(parsed_url.netloc)
    try:
        connection.request("GET", parsed_url.path, headers=HEADERS)
        response = connection.getresponse()
        if response.status == 200:
            with open(save_path, 'wb') as file:
                file.write(decompress_response(response))
            return convert_to_webp(save_path)
    finally:
        connection.close()

def download_image(image_path, save_path):
    try:
        return _download_and_convert_image(image_path, save_path)
    except Exception as err:
        logger.error(f"Error downloading image: {err}")
    return None

def sanitize_column_name(name):
    name = name.replace(" ", "_")
    return f"{name}_col" if name.lower() in reserved_words else name

def extract_icons_and_hrefs(table):
    icon_dict, href_dict = {}, {}
    for row in table.select('tr'):
        item_cells = row.select('td')
        if item_cells:
            item_name = item_cells[0].get_text(strip=True)
            img_tag = item_cells[1].select_one('img')
            href_tag = item_cells[0].select_one('a')
            icon_dict[item_name] = img_tag.get('src', '') if img_tag else ''
            href_dict[item_name] = href_tag.get('href', '') if href_tag else ''
    return icon_dict, href_dict

def pretty_and_save(json_path, output_html_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        if 'parse' in data and 'text' in data['parse'] and '*' in data['parse']['text']:
            soup = BeautifulSoup(data['parse']['text']['*'], 'html.parser')
            with open(output_html_path, 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            os.remove(json_path)
        else:
            logger.error(f'Missing expected keys in JSON data: {data}')
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in pretty_and_save: {e}")

def expand_details_column(df):
    """Expand only 'Type' and 'Slot' from the Details column and rearrange columns."""
    
    # Extract Type and Slot from Details
    def extract_type_slot(details):
        if isinstance(details, str):
            details_dict = json.loads(details)
        else:
            details_dict = details
        # Extract Type and Slot, assign None if not present
        item_type = details_dict.get('Type', None)
        slot = details_dict.get('Slot', None)
        return item_type, slot
    # Apply extraction and create new columns
    df[['Type', 'Slot']] = df['Details'].apply(lambda x: pd.Series(extract_type_slot(x)))
    # Rearrange columns
    desired_columns = [
        'Item', 'Type', 'Slot', 'Page_Name', 'Drops_From', 'Description', 'Details',
        'Stat_Tables', 'Recipe_Table'
    ]
    # Return the DataFrame with the reordered columns
    return df[desired_columns]

def record_page_visit(database, page_name, category):
    """Records a page visit in a database table."""
    try:
        engine = create_engine(f'mysql+mysqlconnector://root:root@localhost/{database}')
        with engine.connect() as connection:
            metadata = MetaData(bind=engine)
            visits_table = Table('visited_pages', metadata,
                                 Column('id', Integer, primary_key=True, autoincrement=True),
                                 Column('page_name', String(255), nullable=False),
                                 Column('category', String(255), nullable=False))
            # Create table if it doesn't exist
            if not visits_table.exists():
                metadata.create_all()
            # Insert a new record for the visited page
            connection.execute(visits_table.insert(), {
                'page_name': page_name,
                'category': category
            })
            logger.info(f"Recorded visit for page {page_name} in category {category}.")
    except SQLAlchemyError as error:
        logger.error(f'Failed to record page visit: {error}')

def create_master_list(category):
    if category != 'Classes':
        csv_files = glob.glob(os.path.join('static', category, '**', 'List_of*.csv'), recursive=True)
        if not csv_files:
            return None
        df_master_list = pd.concat((pd.read_csv(file) for file in csv_files),
                                ignore_index=True)
        if 'type' in df_master_list.columns:
            cols = df_master_list.columns.tolist()
            cols.insert(0, cols.pop(cols.index('type')))
            df_master_list = df_master_list[cols]
    else:
        csv_files = glob.glob(os.path.join('static', category, '**', 'abilities', '*.csv'), recursive=True)

    if not csv_files:
        return None
    
    df_master_list = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)
    df_master_list.to_csv(os.path.join('static', category, f'{category}.csv'), index=False)
    
    return df_master_list

def process_page(row):
    """Process each page and extract information."""
    logger.info(f"Processing {row.page_name} in category {row.category}")
    category, page = row.category, row.page_name
    if not request_json(category, page):
        return

def main():
    df_pages = pd.read_csv('A:/AOC WIki Scraper/page_list.csv')  # Path to your CSV
    database_name = 'gear_fact'
    truncate_table(database_name, 'item_table')
    
    categories = set(df_pages['category'])
    for category in categories:
        df_curr_page = df_pages.loc[df_pages['category'] == category]
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_page, row): row
                for row in df_curr_page.itertuples()
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    future.result()
                    #record_page_visit(database_name, row.page_name, row.category)
                except Exception as exc:
                    logger.error(f'Error processing {row.page_name}: {exc}')
                else:
                    logger.info(f"Successfully processed {row.page_name}")
        df_master_list = create_master_list(category)
        if df_master_list is not None:
            with ThreadPoolExecutor(max_workers=10) as executor:
                item_futures = {
                    executor.submit(fetch_item_info, row.Item, row.Wiki_Link, row.itemType): row
                    for row in df_master_list.itertuples()
                }
                item_info_list = []
                for future in as_completed(item_futures):
                    try:
                        item_info = future.result()
                        if item_info:
                            item_info_list.append(item_info)
                    except Exception as exc:
                        row = item_futures[future]
                        logger.error(f'Error fetching info for {row.Item}: {exc}')
            if item_info_list:
                df_descriptions = pd.DataFrame(item_info_list)
                df_descriptions = expand_details_column(df_descriptions)
                # Check that 'Page_Name' is now in df_descriptions
                create_db_tables(df_descriptions, 'aoc-build-planner', 'item_table')
            create_db_tables(df_master_list, 'aoc-build-planner', category)

if __name__ == "__main__":
    main()
