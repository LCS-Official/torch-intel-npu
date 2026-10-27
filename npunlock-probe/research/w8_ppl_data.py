"""Fetch eight fixed WikiText-2 test slices with the standard library only.

Original dataset: https://huggingface.co/datasets/Salesforce/wikitext
Rows API: https://huggingface.co/docs/dataset-viewer/rows
Downloaded text is temporary evaluation data, not source distribution.
"""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

OUT=Path(__file__).resolve().parent/'results/w8-ppl-generation'
REVISION='b08601e04326c79dfdd32d625aee71d232d685c3'
OFFSETS=(0,500,1000,1500,2000,2500,3000,3500)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',split='test',revision=REVISION,slices=[])
    for offset in OFFSETS:
        query=urllib.parse.urlencode(dict(dataset=report['dataset'],config=report['config'],split='test',offset=offset,length=32))
        url='https://datasets-server.huggingface.co/rows?'+query
        with urllib.request.urlopen(url,timeout=45) as response:
            assert response.headers['x-revision']==REVISION
            raw=response.read()
        data=json.loads(raw)
        assert data['num_rows_total']==4358 and not data['partial'] and len(data['rows'])==32
        assert [row['row_idx'] for row in data['rows']]==list(range(offset,offset+32))
        assert all(not row['truncated_cells'] and type(row['row']['text']) is str for row in data['rows'])
        texts=[row['row']['text'] for row in data['rows']]
        digest=hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        report['slices'].append(dict(offset=offset,url=url,text_sha256=digest,texts=texts))
        print('fetched WikiText test rows',offset,'..',offset+31,flush=True)
    (OUT/'corpus.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS: eight fixed-revision slices; no parquet/dependency installation',flush=True)


if __name__=='__main__':
    main()
