import picker from '@ohos.file.picker'; // 引入文件选择器模块
import http from '@ohos.net.http'; // 引入HTTP模块
import fileio from '@ohos.fileio'; // 引入文件IO模块
import { Base64 } from 'js-base64'; // 引入 js-base64 模块
import request from '@ohos.request';
import { BusinessError } from '@ohos.base';
import fs from '@ohos.file.fs';
import buffer from '@ohos.buffer';

// 定义 data 字段的类型
interface InferenceData {
  image: string; // 检测后的图片路径
  class: string; // 分类结果
}

// 定义后端返回的数据结构
interface InferenceResult {
  code: number; // 响应状态码
  msg: string; // 响应消息
  data: InferenceData; // 数据字段
}

// 定义请求体的类型
interface RequestBody {
  img_64: string; // Base64 编码的图片数据
  model: string; // 模型类型
}

interface SelectOption {
  value: string; // 选项的值
  label: string; // 选项的显示文本
}

@Entry
@Component
struct Index {
  @State originalFile: string = ''; // 默认文件路径
  @State originalFileBase64: string = ''
  @State processedImage: string = ''; // 默认图片路径
  @State inferenceInfo: string = '推理信息将显示在这里'; // 推理信息
  @State isProcessing: boolean = false; // 是否正在推理
  @State selectedModel: string = 'detected'; // 默认选择的模型
  @State modelOptions: SelectOption[] = [
    { value: 'detected', label: '检测模型' },
    { value: 'classification', label: '分类模型' },
  ]; // 模型选项

  // 获取当前选中模型的索引
  getSelectedIndex(): number {
    return this.modelOptions.findIndex(option => option.value === this.selectedModel);
  }

  build() {
    Column() {
      // 文件展示区域，占据剩余空间
      Row() {
        Column() {
          Text('原文件')
            .fontSize(18)
            .fontColor('#333333')
            .margin({ bottom: 10 });
          Image(this.originalFile)
            .width('100%')
            .height('100%')
            .objectFit(ImageFit.Cover)
            .borderRadius(10);
        }
        .width('49%') // 宽度调整为49%
        .height('100%')
        .padding(10)
        .backgroundColor('#F0F8FF') // 淡蓝色
        .borderRadius(10);

        // 中间间距
        Column() {
          // 空白区域，用于留出间距
        }
        .width('2%') // 中间间距宽度为2%
        .height('100%');

        Column() {
          Text('检测图片')
            .fontSize(18)
            .fontColor('#333333')
            .margin({ bottom: 10 });
          Image(this.processedImage)
            .width('100%')
            .height('100%')
            .objectFit(ImageFit.Cover)
            .borderRadius(10);
        }
        .width('49%') // 宽度调整为49%
        .height('100%')
        .padding(10)
        .backgroundColor('#F0FFF0') // 淡绿色
        .borderRadius(10);
      }
      .width('100%')
      .layoutWeight(1) // 占据剩余空间
      .margin({ bottom: 20 });

      // 推理信息区域
      Column() {
        Text('推理信息')
          .fontSize(18)
          .fontColor('#333333')
          .margin({ bottom: 10 });
        Text(this.inferenceInfo)
          .fontSize(16)
          .fontColor('#555555');
      }
      .width('100%')
      .padding(10)
      .backgroundColor('#FFF8DC') // 淡黄色
      .borderRadius(10)
      .margin({ bottom: 20 });

      // 操作区域
      Row() {
        // 文件路径展示（文本框）
        Column() {
          Text('文件路径')
            .fontSize(14)
            .fontColor('#333333')
            .margin({ bottom: 5 });
          Text(this.originalFile)
            .width('100%')
            .height(40)
            .padding(10)
            .backgroundColor('#FFFFFF')
            .borderRadius(5)
            .border({ width: 1, color: '#CCCCCC' })
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis }); // 超出部分显示省略号
        }
        .width('30%') // 文本框宽度
        .margin({ right: 10 });

        // 模型选择下拉列表
        Column() {
          Text('选择模型')
            .fontSize(14)
            .fontColor('#333333')
            .margin({ bottom: 5 });
          Select(this.modelOptions) // 使用 Select 组件
            .selected(this.getSelectedIndex()) // 传入当前选中项的索引
            .onSelect((index: number) => {
              this.selectedModel = this.modelOptions[index].value; // 更新选择的模型
            })
            .width('100%')
            .height(40)
            .backgroundColor('#FFFFFF')
            .borderRadius(5)
            .border({ width: 1, color: '#CCCCCC' });
        }
        .width('30%') // 下拉列表宽度
        .margin({ right: 10 });

        // 按钮区域
        Row() {
          Button('选择文件')
            .onClick(() => {
              this.selectFile();
            })
            .width('30%')
            .backgroundColor('#E6E6FA') // 淡紫色
            .fontColor('#333333');

          Button('开始推理')
            .onClick(() => {
              this.startInference();
            })
            .width('30%')
            .backgroundColor('#E6E6FA') // 淡紫色
            .fontColor('#333333');

          Button('停止')
            .onClick(() => {
              this.stopInference();
            })
            .width('30%')
            .backgroundColor('#E6E6FA') // 淡紫色
            .fontColor('#333333');
        }
        .width('35%') // 按钮区域宽度
        .justifyContent(FlexAlign.SpaceBetween);
      }
      .width('100%')
      .padding(10)
      .backgroundColor('#F5F5F5') // 淡灰色
      .borderRadius(10);
    }
    .width('100%')
    .height('100%')
    .padding(20)
    .backgroundColor('#FFFFFF'); // 白色背景
  }


  // 选择文件
  async selectFile() {
    try {
      // 创建文件选择器
      const documentPicker = new picker.DocumentViewPicker();
      const result = await documentPicker.select({
        maxSelectNumber: 1, // 最多选择1个文件
      });
      console.error(result.toString())
      // 获取选中文件的URI
      if (Array.isArray(result) && result.length > 0) {
        const uri: string = result[0]; // 获取第一个文件的URI
        this.originalFile = uri; // 更新文件路径
        this.inferenceInfo = '文件已选择';
      }

      let file = fs.openSync(result[0], fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE)

      const arrayBuffer = new ArrayBuffer(409600);
      const readLen = fs.readSync(file.fd, arrayBuffer);
      const buf = buffer.from(arrayBuffer, 0, readLen);
      const base64 = buf.toString('base64');
      fs.closeSync(file);
      this.originalFileBase64 = base64
    } catch (err) {
      console.error('选择文件失败:', err);
      this.inferenceInfo = '选择文件失败';
    }
  }


  // 开始推理
  async startInference() {
    if (!this.originalFile) {
      this.inferenceInfo = '请先选择文件';
      return;
    }

    this.isProcessing = true;
    this.inferenceInfo = '推理中...';

    try {

      // 构建请求体，显式指定类型为 RequestBody
      const requestBody: RequestBody = {
        img_64: this.originalFileBase64, // Base64 编码的图片数据
        model: this.selectedModel, // 模型类型
      };



      // 创建HTTP请求
      const httpRequest = http.createHttp();
      const url = 'http://192.168.137.80:5000/upload'; // 替换为你的服务器地址
      const options: http.HttpRequestOptions = {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json',
        },
        extraData: requestBody, // 将请求体转换为 JSON 字符串
        connectTimeout: 60000, // 连接超时时间（60秒）
        readTimeout: 120000, // 读取超时时间（60秒）
      };

      // 发送请求
      const response = await httpRequest.request(url, options);

      // 检查响应状态
      if (response.responseCode === 200) {
        const result: InferenceResult = JSON.parse(response.result.toString()); // 解析响应数据
        if (result.code === 200) {
          if (this.selectedModel === 'detected') {
            this.processedImage = result.data.image; // 更新检测后的图片
            this.inferenceInfo = `检测完成：${result.msg}`;
          } else if (this.selectedModel === 'classification') {
            this.inferenceInfo = `分类结果：${result.data.class}`;
          }
        } else {
          this.inferenceInfo = `推理失败：${result.msg}`;
        }
      } else {
        this.inferenceInfo = `请求失败：${response.responseCode}`;
      }
    } catch (err) {
      // 打印完整的错误信息到终端
      console.error('推理请求失败 - 完整错误信息:', {
        message: err.message, // 错误消息
        stack: err.stack, // 错误堆栈
        name: err.name, // 错误名称
      });
      this.inferenceInfo = `推理请求失败：${err.message || '未知错误'}`; // 显示错误信息
    } finally {
      this.isProcessing = false;
    }
  }

  // 停止推理
  async stopInference() {
    if (this.isProcessing) {
      this.isProcessing = false;
      this.originalFile = '';
      this.processedImage = '';
      this.inferenceInfo = '推理已停止';
    }
  }

  // 读取文件内容并编码为 Base64
  async readFileAsBase64(fileUri: string): Promise<string> {
    let fd: number | undefined; // 声明文件描述符变量
    try {
      // 使用 'r' 模式打开文件，返回文件描述符（number 类型）
      console.log(fileUri);

      const stat1 = await fileio.stat(fileUri);
      fd = fileio.openSync(fileUri)
      console.log(fd.toString())
      const stat = await fileio.stat(fileUri);
      const buffer = new ArrayBuffer(stat.size);

      // 使用文件描述符读取文件内容
      await fileio.read(fd, buffer);

      // 将 ArrayBuffer 转换为 Base64 字符串
      const base64: string = Base64.fromUint8Array(new Uint8Array(buffer));
      return base64;
    } catch (err) {
      console.error('读取文件失败:', err);
      return ''; // 返回默认值（空字符串）
    } finally {
      // 确保文件描述符被关闭
      if (fd !== undefined) {
        await fileio.close(fd);
      }
    }
  }
}