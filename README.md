# Wheat Grains Detector
**For Celiac Disease sufferers**

<span style="display:block;text-align:center">![App in action](ezgif-1-138e5d970f8d.gif)</span>
## Android + OpenCV + Tensorflow Object Detection API
_**Please note:** this is a sidenotes for my own personal use rather than a detailed step-by-step instruction on how to build the project. However an experienced developer should get what's going on._

## Preparing dataset

`LabelImg`/`VoTT`/`LabelBox` for segmentation and annotation are all fine. `LabelImg` is finest.

### Generate TFRecord

```bash
python create_pascal_tf_record_ex.py --annotations_dir dataset/train --label_map_path dataset/label_map.pbtxt --output_path train.record
python create_pascal_tf_record_ex.py --annotations_dir dataset/val --label_map_path dataset/label_map.pbtxt --output_path val.record
```

## Training model

Connect to [Colaboratory](https://colab.research.google.com) and upload _colab/Wheat_Grains_Detector.ipynb_

To upload:
* _colab/ssd_mobilenet_v2_coco.config_
* _dataset/label_map.pbtxt_
* _train.record_
* _val.record_

Will be downloaded:
* _frozen.zip_

## Building Android app

### Install JDK

```bash
sudo apt-get install openjdk-8-jdk-headless
```
or

Download [Java SE Development Kit 8](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) and unpack to _`<JDK_DIR>`_. _jdk-8u181-linux-x64.tar.gz_ was used.
```bash
export PATH=$PATH:<JDK_DIR>/bin/
```

### Install Android SDK

Download [SDK tools](https://developer.android.com/studio/) and unpack to _`<ANDROID_SDK_DIR>`_. _sdk-tools-linux-4333796.zip_ was used.
```bash
export ANDROID_SDK_HOME=<ANDROID_SDK_DIR>

export PATH=$PATH:$ANDROID_SDK_HOME/tools/bin/

sdkmanager "build-tools;28.0.2"
export PATH=$PATH:$ANDROID_SDK_HOME/build-tools/28.0.2/

sdkmanager "platforms;android-21"

sudo apt-get install adb
```
Connect `adb` to a device.

### Build OpenCV library

Download [Android pack](https://opencv.org/releases.html) and unpack to _`<OPENCV_SDK_DIR>`_. _opencv-3.4.3-android-sdk.zip_ was used.
```bash
export OPENCV_SDK_JAVA=<OPENCV_SDK_DIR>/sdk/java/
pushd $OPENCV_SDK_JAVA
mkdir -p build/gen/ build/obj/
aapt package -m -J build/gen/ -M AndroidManifest.xml -S res/ -I $ANDROID_SDK_HOME/platforms/android-21/android.jar
aidl -obuild/gen/ src/org/opencv/engine/OpenCVEngineInterface.aidl
```
create _build/gen/BuildConfig.java_ with the following content:
```java
package org.opencv;

public final class BuildConfig {
    public static final boolean DEBUG = Boolean.parseBoolean("false");
}
```
```bash
javac -d build/obj/ -bootclasspath $ANDROID_SDK_HOME/platforms/android-21/android.jar build/gen/BuildConfig.java build/gen/org/opencv/R.java build/gen/org/opencv/*/*.java src/org/opencv/*/*.java
aapt package -F build/opencv.jar -M AndroidManifest.xml -S res/ -I $ANDROID_SDK_HOME/platforms/android-21/android.jar build/obj/
popd
```

### Build app

```bash
pushd android
mkdir -p build/gen/ build/obj/ build/bin/lib/ build/bin/assets/
aapt package -m -J build/gen/ -M AndroidManifest.xml -S $OPENCV_SDK_JAVA/res/ -S res/ -I $ANDROID_SDK_HOME/platforms/android-21/android.jar
javac -d build/obj/ -bootclasspath $ANDROID_SDK_HOME/platforms/android-21/android.jar -classpath $OPENCV_SDK_JAVA/build/opencv.jar build/gen/com/github/failure_to_thrive/wheat_grains_detector/R.java src/com/github/failure_to_thrive/wheat_grains_detector/MainActivity.java
dx --dex --output=build/bin/classes.dex $OPENCV_SDK_JAVA/build/opencv.jar build/obj/
```
copy _`<OPENCV_SDK_DIR>`/sdk/native/libs/armeabi-v7a/_ to _build/bin/lib/_  
copy _`<OPENCV_SDK_DIR>`/sdk/native/libs/arm64-v8a/_ to _build/bin/lib/_  
unpack _frozen.zip_ to _build/bin/assets/_
```bash
aapt package -F build/WheatGrainsDetector.unaligned.apk -M AndroidManifest.xml -S $OPENCV_SDK_JAVA/res/ -S res/ -I $ANDROID_SDK_HOME/platforms/android-21/android.jar build/bin/
apksigner sign --key <key> --cert <cert> build/WheatGrainsDetector.unaligned.apk
zipalign -p 4 build/WheatGrainsDetector.unaligned.apk build/WheatGrainsDetector.apk
adb install build/WheatGrainsDetector.apk
popd
```
```bash
adb logcat WGD:* *:S
```

That's all!
