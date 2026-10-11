"""Change one binding contract to verify safe hardware fallback in a fresh process."""
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parent/'results/startup-ir-cache'
path=next((OUT/'candidate-cache').glob('*.npc'))
with zipfile.ZipFile(path) as archive:
    manifest=json.loads(archive.read('manifest.json'));blob=archive.read('graph.blob')
manifest['io_contract']['outputs'][0]['shape']=[1,1,576]
with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as archive:
    archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('graph.blob',blob)
(OUT/'corrupted-entry.json').write_text(json.dumps(dict(key=manifest['key'],field='io_contract.outputs[0].shape')))
print('One prefill contract deliberately corrupted; compiled blob unchanged')
