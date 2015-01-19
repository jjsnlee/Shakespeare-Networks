
def main():
	import shakespeare.clusters_runner as scr
	import shakespeare.clusters_documents as scd
	import shakespeare.clusters as sc

	c=scd.EEBODocumentsCtxt()
	titles,itfn=c.get_doc_content(getmax=10)
	ldac=sc.LDAContext(titles,itfn(),stopwds=scr._get_stopwords())
	ldar=sc.LDAResult('eebo-test', ldac, ntopics=10, npasses=20)
	ldar.termite_data.data_for_client()

if __name__ == '__main__':
	main()
