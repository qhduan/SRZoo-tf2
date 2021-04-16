import React from 'react'

const jpeg = require('jpeg-js')

const myWorker = new Worker('worker.js')
myWorker.onmessage = (e) => {
    console.log('Message received from worker', e.data)
    drawImage(e.data, false, 'Super Resolution Image', '#img2')
}


function fixPixel(v) {
    v = Math.round(v)
    v = Math.max(0, v)
    v = Math.min(v, 255)
    return v
}

async function drawImage(obj, mul=false, text='', place='#img1') {
    const height = obj.height
    const width = obj.width
    const data = obj.data

    const container = document.createElement('div')

    const canvas = document.createElement('canvas')

    canvas.width = width
    canvas.height = height
    canvas.style.width = '50vw'
    const context = canvas.getContext('2d')
    const imageData = new ImageData(width, height)
    const buffer = new Uint8ClampedArray(width * height * 4)
    let iAdd = 3
    if (data.length === width * height * 4) {
        iAdd = 4
    }
    let i = 0
    for(let y = 0; y < height; y++) {
        for(let x = 0; x < width; x++) {
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
    if (mul) {
        // container.style.width = `${width * 2}px`
        // container.style.height = `${height * 2}px`
        // canvas.style.transform = 'scale(2)'
        // canvas.style.transformOrigin = '0px 0px'
        // container.style.width = '50vw'
    }
    container.innerHTML = `<p>${text}</p>`
    container.appendChild(canvas)
    document.querySelector(place).appendChild(container)
}


class App extends React.Component {

    upload = (e) => {
        const file = e.target.files[0]
        const reader = new FileReader()
        reader.onload = async (event) => {
            const buf = event.target.result
            const pixels = jpeg.decode(buf, true)
            // console.log(pixels)
            drawImage(pixels, true, 'Original Image')
            myWorker.postMessage(pixels)
        }
        reader.onerror = (err) => {
            console.error(err)
        }
        reader.readAsArrayBuffer(file)
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

export default App
