"""Install the official portable Graphviz build inside this project."""
import hashlib,json,urllib.request,zipfile
from setup_local_runtime import ROOT,fetch

VERSION='16.1.0'
NAME=f'windows_10_cmake_Release_Graphviz-{VERSION}-win64.zip'
BASE=f'https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/{VERSION}'


def main():
    with urllib.request.urlopen(f'{BASE}/{NAME}.sha256',timeout=60) as response:
        digest=response.read().decode().split()[0]
    if len(digest)!=64 or any(c not in '0123456789abcdefABCDEF' for c in digest):
        raise ValueError('Invalid official checksum')
    archive=ROOT/'data'/'downloads'/NAME
    fetch(f'{BASE}/{NAME}',archive,digest.lower())
    target=ROOT/'tools'/'graphviz';target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (target/item.filename).resolve().is_relative_to(target.resolve()):
                raise ValueError('Unsafe Graphviz archive path')
        bundle.extractall(target)
    dot=next(target.rglob('dot.exe'))
    manifest={'version':VERSION,'url':f'{BASE}/{NAME}','archive_sha256':digest.lower(),
        'dot':str(dot.relative_to(ROOT)),'dot_sha256':hashlib.sha256(dot.read_bytes()).hexdigest()}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Portable Graphviz ready:',dot)


if __name__=='__main__':main()
