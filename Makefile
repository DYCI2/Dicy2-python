PYINSTALLER_PATH = pyinstaller
PYINSTALLER_TARGET = dyci2/server.py
PYINSTALLER_TARGET_NAME = dyci2_server
APP_PATH = dist/dyci2_server.app
DMG_PATH = dist/dyci2_server.dmg



pyinstaller:
	@echo "\033[1m####### Building server binary with pyinstaller ########\033[0m"
	$(PYINSTALLER_PATH) $(PYINSTALLER_TARGET) \
		--clean \
		--noconfirm \
		--onedir \
		--noconsole \
		--name $(PYINSTALLER_TARGET_NAME) \
		--exclude-module matplotlib \
		--exclude-module PyQt5 \
		--hidden-import="cmath" \
		--codesign-identity="Developer ID Application: INST RECHER COORD ACOUST MUSICALE" \
		--osx-entitlements-file="codesign/dyci2.entitlements"

notarize:
	hdiutil create "$(DMG_PATH)" -fs HFS+ -srcfolder "${APP_PATH}" -ov
	xcrun notarytool submit "$(DMG_PATH)" --keychain-profile "repmus" --wait
	xcrun stapler staple "$(APP_PATH)"


dmg:
	hdiutil create "$(DMG_PATH)" -fs HFS+ -srcfolder "$(APP_PATH)" -ov


clean:
	rm -rf build


clean-all:
	rm -rf build dist
