import React from 'react'

const tf = require('@tensorflow/tfjs')
const jpeg = require('jpeg-js')

let model = null

async function loadModel() {
    model = await tf.loadGraphModel('carn_x2_js/model.json')
    console.log('model loaded')
}

loadModel()

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


function fixPixel(v) {
    v = Math.round(v)
    v = Math.max(0, v)
    v = Math.min(v, 255)
    return v
}


async function drawImage(tensor, mul=false) {
    const height = tensor.shape[1]
    const width = tensor.shape[2]
    var canvas2 = document.createElement('canvas');
    canvas2.width = width;
    canvas2.height = height;
    var context2 = canvas2.getContext('2d');
    const imageData = new ImageData(width, height);
    const buffer = new Uint8ClampedArray(width * height * 4)
    const data = tensor.dataSync();
    var i = 0;
    for(var y = 0; y < height; y++) {
        for(var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4;
            buffer[pos + 0] = fixPixel(data[i])
            buffer[pos + 1] = fixPixel(data[i + 1])
            buffer[pos + 2] = fixPixel(data[i + 2])
            buffer[pos + 3] = 255
            i += 3
        }
    }
    imageData.data.set(buffer)
    context2.putImageData(imageData, 0, 0)
    if (mul) {
        canvas2.style.transform = 'scale(2)'
        canvas2.style.transformOrigin = '0'
    }
    document.querySelector('body').appendChild(canvas2)
}


class App extends React.Component {

    upload = (e) => {
        const file = e.target.files[0];

        const reader = new FileReader();

        reader.onload = async (event) => {
            const buf = event.target.result
            const pixels = jpeg.decode(buf, true)
            console.log(pixels)
            let input = await imageToInput(pixels, 3)
            console.log(input)

            console.log('Draw Original')
            drawImage(input, true)

            // carn setting
            let pixel_max_one = true
            let channel_first = true

            if (channel_first) {
                input = await tf.transpose(input, [0, 3, 1, 2])
            }
            if (pixel_max_one) {
                input = tf.div(input, 255.0)
            }
    
            let out = await model.executeAsync(input)

            if (channel_first) {
                out = tf.transpose(out, [0, 2, 3, 1])
            }
            if (pixel_max_one) {
                out = tf.mul(out, 255.0)
            }

            console.log('Draw Better')
            drawImage(out)
        };

        reader.onerror = (err) => {
            console.error(err);
        };

        reader.readAsArrayBuffer(file);
    }

    render() {
        return (
            <div className="App">
                <h3>X2</h3>
                <input type='file' onChange={this.upload} />
            </div>
        )
    }
}

export default App;
