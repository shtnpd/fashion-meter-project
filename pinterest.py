from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def download_images_from_pinterest(style):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    url = f"https://www.pinterest.com/search/pins/?q={style.replace(' ', '%20')}"
    driver.get(url)
    images = driver.find_elements(By.CSS_SELECTOR, "img")
    urls = [img.get_attribute("src") for img in images if img.get_attribute("src") is not None]
    driver.quit()
    return urls

style = "cottage core"
images = download_images_from_pinterest(style)
print(images[:10])
