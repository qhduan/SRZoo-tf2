// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// All of the Node.js APIs are available in this process.

const path = require('path')

const tf = require('@tensorflow/tfjs-node')
const jpeg = require('jpeg-js')

const nodeConsole = require('console');
const myConsole = new nodeConsole.Console(process.stdout, process.stderr);

function setLoading(show) {
  const e = document.querySelector('#loading')
  if (show) {
    e.style.visibility = 'visible'
  } else {
    e.style.visibility = 'hidden'
  }
}


const imageByteArray = async (image, numChannels) => {
    const pixels = image.data
    const numPixels = image.width * image.height;
    const values = new Int32Array(numPixels * numChannels);

    for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
            values[i * numChannels + channel] = pixels[i * 4 + channel];
        }
    }

    return values
}

const imageToInput = async (image, numChannels) => {
    const values = await imageByteArray(image, numChannels)
    const outShape = [1, image.height, image.width, numChannels]
    const input = tf.tensor4d(values, outShape, 'float32')

    return input
}


async function scale(image, modelPath='eusr_x8') {
    // Change this
    // const imagePath = 'input.jpg'
    modelPath = path.join(__dirname, 'extraResources', modelPath)
    myConsole.log('start scale model', modelPath)
    // const modelPath = `./models/${model}`

    // Different model use different format
    // NCHW vs NHWC
    let pixel_max_one = false
    if (modelPath.match(/carn|dbpn|esrgan|frsr|natsr|rrdb/)) {
        pixel_max_one = true
    }

    // channel_first = True
    // [0, 1] vs [0, 255]
    let channel_first = true
    if (modelPath.match(/4pp_eusr|eusr|frsr|natsr/)) {
        channel_first = false
    }

    myConsole.log('start make model input')

    try {
        let input = await imageToInput(image, 3)

        if (channel_first) {
            input = tf.transpose(input, [0, 3, 1, 2])
        }
        if (pixel_max_one) {
            input = input / 255.0
        }

        console.log('read model')

        const model = await tf.node.loadSavedModel(modelPath)
        const out = model.predict(input)

        // let p = await out.arraySync()
        let p = out

        if (channel_first) {
            p = tf.transpose(p, [0, 2, 3, 1])
        }
        if (pixel_max_one) {
            p = p * 255.0
        }

        const height = p.shape[1]
        const width = p.shape[2]
        const bufData = await p.buffer()

        const frameData = new Buffer(width * height * 4)
        for (let i = 0; i < height * width; i++) {
            for (let j = 0; j < 3; j++) {
                let v = bufData.values[i * 3 + j]
                v = Math.round(v)
                v = Math.max(0, v)
                v = Math.min(v, 255)
                frameData[i * 4 + j] = v
            }
            frameData[i * 4 + 3] = 0xff
        }

        myConsole.log('draw model result')

        drawImage({
          data: frameData,
          height,
          width,
          scaled: true
        }, 'Fixed Image, right click to save')

        drawImage({
          data: frameData,
          height,
          width,
          keep: true
        }, 'Scaled Image, right click to save')

        setLoading(false)

    } catch (e) {
        myConsole.error(e)
    }
}



function fixPixel(v) {
  v = Math.round(v)
  v = Math.max(0, v)
  v = Math.min(v, 255)
  return v
}


async function drawImage(obj, text = '') {

  const height = obj.height
  const width = obj.width
  const data = obj.data

  const container = document.createElement('div')

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  const imageData = new ImageData(width, height)
  const buffer = new Uint8ClampedArray(width * height * 4)
  let iAdd = 3
  myConsole.log('render', obj)
  if (data.length === width * height * 4) {
    iAdd = 4
  }
  let i = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pos = (y * width + x) * 4
      buffer[pos + 0] = fixPixel(data[i + 0])
      buffer[pos + 1] = fixPixel(data[i + 1])
      buffer[pos + 2] = fixPixel(data[i + 2])
      buffer[pos + 3] = 255
      i += iAdd
    }
  }
  imageData.data.set(buffer)
  context.putImageData(imageData, 0, 0)
  container.innerHTML = `<p>${text}</p>`

  const image = new Image()
  image.src = canvas.toDataURL("image/png")
  if (obj.keep) {
    // do nothing
  } else if (obj.scaled) {
    image.style.maxWidth = `${Math.min(400, width / 8)}px`
  } else {
    image.style.maxWidth = '400px'
  }
  container.appendChild(image)

  document.querySelector('#content').appendChild(container)
}

document.querySelector('#clean').addEventListener('click', () => {
  document.querySelector('#content').innerHTML = ''
})

document.querySelector('input').addEventListener('change', async (e) => {
  myConsole.log('start read file', e.target.files[0])
  const file = e.target.files[0]
  const reader = new FileReader()
  reader.onload = async (event) => {
    const buf = event.target.result
    const pixels = jpeg.decode(buf, true)
    myConsole.log(pixels)
    drawImage(pixels, 'Original Image')
    setLoading(true)
    setTimeout(() => scale(pixels), 10)
  }
  reader.onerror = (err) => {
    myConsole.error(err)
  }
  reader.readAsArrayBuffer(file)
}, true)
