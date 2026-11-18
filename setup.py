"""Install frozen modules and parameter files in one private package tree."""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

# Frozen model code opens profiles relative to the parent of its source folder.
# Preserve that layout within operator_model, without adding top-level packages.
class BuildModelPackage(build_py):
    def get_module_outfile(self, build_dir, package, module):
        if package == ['operator_model'] and module != '__init__':
            package = [*package, '_engine']
        return super().get_module_outfile(build_dir, package, module)

    def run(self):
        super().run()
        if self.editable_mode:
            return
        package_dir = Path(self.build_lib) / 'operator_model'
        # A static path extension keeps public imports and python -m unchanged.
        init = Path('operator_model/__init__.py').read_text(encoding='utf-8')
        prefix = "from pathlib import Path as _Path\n__path__.append(str(_Path(__file__).with_name('_engine')))\n"
        # Keep the package docstring and imports in their usual positions.
        docstring, body = init.split('\n', 1)
        (package_dir / '__init__.py').write_text(docstring + '\n' + prefix + body, encoding='utf-8')
        for line in Path('MANIFEST.in').read_text(encoding='utf-8').splitlines():
            if line.startswith('include results/') or line.startswith('include configs/'):
                source = line.removeprefix('include ')
                dest = package_dir / source
                self.mkpath(str(dest.parent))
                self.copy_file(source, str(dest))


setup(cmdclass={'build_py': BuildModelPackage})
