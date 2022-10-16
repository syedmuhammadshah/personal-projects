import scrapy
class AmazonSamsungSpider(scrapy.Spider):
    name = 'amazon_samsung'
    allowed_domains = ['https://www.amazon.com']
    URL="https://www.amazon.com/Samsung-Bluetooth-Unlocked-Advanced-monitoring/product-reviews/B089DMR9LD/ref=cm_cr_arp_d_paging_btm_next_2?pageNumber="
    start_urls = []
    
    for i in range(1,5):
        start_urls.append(URL+str(i))
    

    def parse(self, response):
        
         page_div = response.css('#cm_cr-review_list')

         customer = page_div.css('.a-profile-content')

         comments = page_div.css('.review-text')
         count = 0

        
         for i in customer:
            yield{'customer': ''.join(i.xpath('.//text()').extract()),
                  'customer review': ''.join(comments[count].xpath('.//text()').extract())
                 }
            count=count+1

            
#customer_review-R1PYNWQWS04NSK > div:nth-child(1) > a > div.a-profile-content > span
#customer_review-R1PYNWQWS04NSK > div.a-row.a-spacing-small.review-data > span > span 


