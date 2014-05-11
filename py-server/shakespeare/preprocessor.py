from BeautifulSoup import BeautifulSoup
from os.path import join
import helper
import re

def transform_chekhov_plays():
    basedir = join(helper.get_root_dir(), 'data/chekhov')
    full_file = join(basedir, '_gutenberg_plays_second_series.html')
    with open(full_file) as f:
        all_plays = ''.join(f.readlines())

    soup = BeautifulSoup(all_plays)
    plays = soup.findAll('div', {'class':'play'})
    
    templ = '''<html>
<head>
<title>%s</title>
</head>
<body>
<h2>%s</h2>
%s
</body>
</html>
'''

    for play in plays:
        hdrs = play.findAll('h2')
        title = hdrs[0].text
        print 'title:', title
        content = []
        lineno = [1]
        if len(hdrs) > 1:
            act_hdrs = hdrs[1:]
            acts = parse_chekhov_play(act_hdrs)
            hard_coded_scene = 1
            for i, act in enumerate(acts):
                act_nm = act_hdrs[i].text
                content.append('<h3>%s</h3>'%act_nm)
                for line in act:
                    content.append(write_line(line, hard_coded_scene, i+1, lineno))
        else:
            for line in hdrs[0].findNextSiblings(['p']):
                content.append(write_line(line.text, 1, 1, lineno))

        fname = title+'.html'
        with open(join(basedir, fname), 'w') as fh:
            html = templ % (title, title, '\n'.join(content))
            fh.write(html.encode('utf8'))

char_line_re = re.compile(r'^([A-Z][^.]+)\.(.+)$', re.S)

def write_line(line, actno, sceneno, lineno):
    line = line.replace('\n', '<br>\n')
    m = char_line_re.match(line) 
    #if char_line_re.match(line, re.S):
    if m:
        speaker, dialogue = m.groups()
        out = '<a name=%d.%d.%d><b>%s.</b>%s</a><br>' % (actno, sceneno, lineno[0], speaker, dialogue)
        #out = '<a name=%d.%d.%d>%s</a><br>' % (actno, sceneno, lineno[0], line)
        lineno[0] += 1
    else:
        out = '<p>%s</p>' % line
    return out

def parse_chekhov_play(act_hdrs):
    acts = []
    nacts = len(act_hdrs)
    for i in range(nacts):
        sc = act_hdrs[i]
        next_sc = act_hdrs[min(i+1, nacts-1)]
        acts.append([])
        for para in sc.findNextSiblings(['p', 'h2']):
            if para == next_sc:
                break
            acts[i].append(para.text)
    return acts
