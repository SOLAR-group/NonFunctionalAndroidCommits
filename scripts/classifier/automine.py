import git
import shutil
import os
import subprocess

repolist = ["https://github.com/PaperAirplane-Dev-Team/GigaGet","https://github.com/shagr4th/droid48.git","https://github.com/bparmentier/OpenBikeSharing","https://github.com/valerio-bozzolan/AcrylicPaint.git","https://github.com/ligi/BLExplorer.git","https://github.com/hugomg/DailyPill","https://github.com/SecUSo/privacy-friendly-passwordgenerator","https://github.com/sadr0b0t/yashlang","https://github.com/NHellFire/KernelAdiutor","https://github.com/jatwigg/cmus-android-remote","https://github.com/DevelopFreedom/logmein-android.git","https://github.com/TheAkki/Synctool.git","https://github.com/Gittner/OSMBugs","https://github.com/JohnMH/BoogDroid","https://github.com/ushahidi/SMSSync.git","https://github.com/vackosar/search-based-launcher-v2.git","https://github.com/Domi04151309/AlwaysOn","https://github.com/arduia/ProExpense","https://github.com/cernekee/EasyToken","https://github.com/jeffboody/gears2","https://github.com/experiment322/controlloid-client","https://github.com/foxykeep/lifecounter","https://github.com/wulkanowy/wulkanowy","https://github.com/AnySoftKeyboard/LanguagePack.git", "https://github.com/AndreMiras/QrScan.git","https://github.com/lllllT/AtmosphereLogger.git","https://github.com/stephanepoinsart/votar","https://github.com/ksksue/Android-USB-Serial-Monitor-Lite.git","https://github.com/gsantner/dandelion","https://github.com/bernaferrari/ChangeDetection","https://github.com/SecUSo/privacy-friendly-netmonitor/","https://github.com/john-tornblom/TVHGuide.git","https://github.com/Governikus/AusweisApp2","https://github.com/xorum-io/open_money_tracker","https://github.com/apps4av/avare", "https://github.com/koelleChristian/trickytripper.git","https://github.com/GabrielTavernini/Covid19Stats","https://github.com/yaa110/Memento","https://github.com/Alikaraki95/Getoffyourphone.git","https://github.com/gpodder/GpodRoid.git","https://github.com/jkennethcarino/AnkiEditor.git","https://github.com/queler/holokenmod","https://github.com/friimaind/pi-hole-droid","https://github.com/linuxtage/glt-companion.git","https://github.com/flackbash/AudioAnchor","https://github.com/michaelkourlas/voipms-sms-client","https://github.com/harleensahni/media-button-router","https://github.com/flyve-mdm/android-inventory-agent.git", "https://github.com/idunnololz/igo/", "https://github.com/anthonycr/Lightning-Browser.git","https://github.com/quaap/AudioMeter","https://github.com/SecUSo/privacy-friendly-reckoning-skills.git","https://github.com/zamojski/TowerCollector","https://github.com/koreader/koreader.git","https://github.com/jackpal/glesquake","https://github.com/sultanahamer/PermissionsManager","https://github.com/scoute-dich/Weather","https://github.com/lgallard/qBittorrent-Client/","https://github.com/openfoodfacts/openfoodfacts-androidapp","https://github.com/ZeusLN/zeus.git","https://github.com/chelovek84/mLauncher","https://github.com/onyxbits/listmyaps","https://github.com/namlit/siteswap_generator","https://github.com/marunjar/anewjkuapp.git","https://github.com/AnySoftKeyboard/LanguagePack.git","https://github.com/taky/effy.git","https://github.com/iAcn/MBEStyle","https://github.com/Waboodoo/HTTP-Shortcuts.git","https://github.com/yellowbluesky/PixivforMuzei3","https://github.com/zaki50/MemoPad.git","https://github.com/aaronjwood/PortAuthority","https://github.com/sytolk/TaxiAndroidOpen","https://github.com/eugmes/headingcalculator.git","https://github.com/xyzz/openmw-android.git","https://github.com/btmura/rbb.git","https://github.com/rosuH/EasyWatermark","https://github.com/ushahidi/Ushahidi_Android.git","https://github.com/VelbazhdSoftwareLLC/vitosha-blackjack.git","https://github.com/jyio/botbrew-gui.git","https://github.com/MBach/AutoAirplaneMode"]

rPath = os.getcwd() +  "/repos"
count = 0
for repo in repolist:
    allComs = open(os.getcwd() + "/commits.txt", "w")
    allComs.write(repo)
    allComs.write("\n")
    if not os.path.isdir(rPath):
        os.mkdir(rPath)
    try:
        cloned_repo = git.Git(rPath).clone(repo)
    except Exception as e:
        print(e)
        continue
    for f in os.scandir(rPath):
        fgit = git.Git(f.path)
        print(f)
        log = fgit.log()
        shutil.rmtree(f.path)
        allComs.write(log.encode('utf-8', errors="ignore").decode('utf-8'))
        allComs.write("\n\n")
    count += 1
    allComs.close()
    print(subprocess.getoutput("python3 reproduce.py"))

