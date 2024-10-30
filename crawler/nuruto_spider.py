from bs4 import BeautifulSoup
import scrapy

class NarutoSpiderSpider(scrapy.Spider):
    name = "nuruto_spider"
    allowed_domains = ["naruto.fandom.com"]
    start_urls = ["https://naruto.fandom.com/wiki/Special:BrowseData/Jutsu?limit=250&offset=0&_cat=Jutsu"]

    def parse(self, response):
        # Extract jutsu links from the current page
        fandoms = response.css('.smw-columnlist-container a::attr(href)').extract()
        for move in fandoms:
            url = response.urljoin(move)
            yield scrapy.Request(url, callback=self.parse_page)

        # Follow pagination links
        next_pages = response.css('.mw-pager-navigation-bar .mw-next-link::attr(href)').extract()
        for page in next_pages:
            next_page_url = response.urljoin(page)
            yield scrapy.Request(next_page_url, callback=self.parse)

    def parse_page(self, response):
        # Extract details from each jutsu page
        jutsu_title = response.css('#firstHeading > span::text').get()
        justsu_types = ['Genjutsu', 'Ninjutsu', 'Taijutsu', "Kenjutsu"]
        justsus = response.css('.pi-data-value.pi-font a::attr(title)').getall()
        jutsu_classification = [i for i in justsu_types if i in justsus] or ['Ninjutsu']
        selector = response.css('div.mw-parser-output').get()
        soup = BeautifulSoup(selector, 'lxml').find('div')

        # Remove the aside section if it exists
        aside = soup.find('aside')
        if aside:
            aside.decompose()

        jutsu_description = soup.get_text(strip=True)
        jutsu_description = jutsu_description.split('Trivia')[0].strip()

        yield {
            'jutsu_title': jutsu_title,
            'jutsu_type': jutsu_classification,
            'jutsu_description': jutsu_description
        }
