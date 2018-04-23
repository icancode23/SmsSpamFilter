## functions we use to extract features out of text messages
import re

def has_url(message):
	url=re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', message)
	if len(url)>0:
		return 1
	else:
		return 0

def has_mathematical_symbols(message):
	symbol=re.findall('[/+-/*()]', message)

	if len(symbol)>0:
		return 1
	else:
		return 0

def has_dots(message):
	dots=re.findall('[\.]', message)
	print dots


print has_dots("nipun.arora.23")