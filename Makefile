# To build a distributable version of the dyci2_server.py , the full procedure is the following:
#
# ```shell
# make pyinstaller
# make notarize
# make dmg
# ```
#
# Again, you will need to adapt the `pyinstaller` command to correspond to your own codesigning identity.
#
# You will also need to adapt the `xcrun notarytool` line of the `make notarize` command to correspond to your
#   own app-specific password generated from your [Apple Developer Account](https://appleid.apple.com/account/).
#   For more info on this step, see `xcrun notarytool --help`
#
# Note that PyInstaller does not support cross-compilation and that applications built with
#   PyInstaller are generally forward-compatible but not backward-compatible with earlier MacOS versions
#   than the system it was built on. An application built on an Intel Mac is also generally compatible with M1 (or M2)
#   macs but not vice versa. It is therefore recommended to build the app on the oldest OS that should be supported.
#
# Our normal procedure has been to:
# * run `make pyinstaller` on High Sierra (MacOS 10.13) to support every OS from High Sierra to Monterey
# * run `make notarize` on Big Sur (MacOS 11.0) or later (since `notarytool` was introduced with MacOS 11)
#



PYINSTALLER_PATH = pyinstaller
PYINSTALLER_TARGET = dyci2_server.py
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
