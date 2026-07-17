"""Spider NL2SQL dataset analysis for star-schema prevalence study."""

from talk2metadata.analysis.spider.analyzer import SpiderAnalyzer
from talk2metadata.analysis.spider.downloader import SpiderDownloader
from talk2metadata.analysis.spider.models import DatabaseSchema, StarSchemaReport

__all__ = ["SpiderDownloader", "SpiderAnalyzer", "DatabaseSchema", "StarSchemaReport"]
