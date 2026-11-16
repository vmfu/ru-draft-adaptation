@echo off
"C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe" install --quiet --norestart --nocache --installPath G:\VS2022BT --productId Microsoft.VisualStudio.Product.BuildTools --channelId VisualStudio.17.Release --channelUri "https://aka.ms/vs/17/release/channel" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621
exit /b %ERRORLEVEL%
