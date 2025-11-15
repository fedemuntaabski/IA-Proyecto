#!/usr/bin/env python3
"""
Script to compile translation files
"""

import polib
import os

def compile_translations():
    """Compile all PO files to MO files"""
    locales = ['es', 'en']

    for locale in locales:
        po_path = f'locale/{locale}/LC_MESSAGES/messages.po'
        mo_path = f'locale/{locale}/LC_MESSAGES/messages.mo'

        if os.path.exists(po_path):
            try:
                po = polib.pofile(po_path)
                po.save_as_mofile(mo_path)
                print(f"✓ Compiled {po_path} -> {mo_path}")
            except Exception as e:
                print(f"✗ Error compiling {po_path}: {e}")
        else:
            print(f"✗ PO file not found: {po_path}")

if __name__ == '__main__':
    compile_translations()