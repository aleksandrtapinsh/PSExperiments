from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import requests
import os
import time

base_url = "https://replay.pokemonshowdown.com"
url = f"{base_url}/?format=gen9randombattle"
options = Options()
options.add_argument("--headless")  # runs without opening browser

driver = webdriver.Chrome(options=options)
driver.get(url)

pages = 5

while pages > 0:
    time.sleep(2)  # wait for page to load
    print(f"Scraping: {driver.current_url}")
    links = driver.find_elements(By.CSS_SELECTOR, "ul.linklist a")

    log_files = []
    for link in links:
        new_link = new_link = link.get_attribute("href")
        log_files.append(f'{new_link.split("/")[-1]}.log')

    for log in log_files:
        try:
            download_url = f"{base_url}/{log}"
            data = requests.get(download_url).text
            # Get the directory where scraper.py is located
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            # Create path to logs folder
            log_dir = os.path.join(BASE_DIR, "logs")

            # Create a log file path
            log_file = os.path.join(log_dir, log)
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception as e:
            print(f"Error downloading {log}: {e}")
    # get next page
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, "p.pagelink:last-child a.button")
        next_button.click()
        pages -= 1
    except Exception as e:
        print("No more pages to scrape.")
        break
driver.quit()