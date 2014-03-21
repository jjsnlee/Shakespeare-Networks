import helper
from os.path import join

def transform(play_html):
    from BeautifulSoup import BeautifulSoup
    soup = BeautifulSoup(play_html)
    toc_content = soup.findAll('p', {'class':'toc'})
    for toc in toc_content:
        links = toc.findAll('a')
        if links:
            for link in links:
                print link.text, link['href']
    #print toc_content
    
    content = soup.findAll('pre', {'xml:space':'preserve'})
    print content

def generate_shakespeare_files(gen_imgs=False, gen_md=False, gen_lines=False):
    from plays_n_graphs import get_plays_ctx, init_play
    from shakespeare_pages import PlayJSONMetadataEncoder, PlayJSONContentEncoder
    import json
    data_ctx = get_plays_ctx()
    plays = data_ctx.plays
    
    basedir = helper.get_dynamic_rootdir()
    helper.ensure_path(join(basedir, 'json'))
    for play_alias, _ in plays:
        print play_alias
        if gen_md or gen_lines:
            play = init_play(play_alias, False)
            
            if gen_md:
                json_rslt = json.dumps(play, ensure_ascii=False, 
                                       cls=PlayJSONMetadataEncoder, indent=True)
                fname = join(basedir, 'json', play_alias+'.json') 
                with open(fname, 'w') as fh:
                    fh.write(json_rslt)

            if gen_lines:
                json_rslt = json.dumps(play, ensure_ascii=False, 
                                       cls=PlayJSONContentEncoder, indent=True)
                fname = join(basedir, 'json', play_alias+'_content.json') 
                with open(fname, 'w') as fh:
                    fh.write(json_rslt)
        
        if gen_imgs:
            init_play(play_alias, True)

def main_shakespeare_batch():
    generate_shakespeare_files()

def main_gutenberg():
    rootdir = helper.get_root_dir()
    fname = join(rootdir, 'data/marlowe', 'tamburlaine_1.html')
    with open(fname, 'r') as fh:
        play_content = fh.read()
    transform(play_content)

if (__name__=="__main__"):
    #main_gutenberg()
    main_shakespeare_batch()
