import asyncio
import schedule
import time

from scrape.scrape import get_data_via_api, scrape

bbc = "https://www.bbc.com/"

def run_fetch_api():
    asyncio.run(get_data_via_api())

def run_scrape():
    asyncio.run(
        scrape(
            bbc,
            section_container='div',
            inner_section='sc-cd6075cf-0 cJhFtM',
            element='p',
            id=False
        )
    )

schedule.every(24).hours.do(run_fetch_api)
schedule.every(2).hours.do(run_scrape)


run_fetch_api()
run_scrape()

while True:
    schedule.run_pending()
    time.sleep(60)