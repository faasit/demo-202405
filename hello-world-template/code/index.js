const { CloudEventCreators } = require('./gen/events')

const { createFunction, createExports } = require('@faasit/runtime')

const handler = createFunction(async (frt) => {
  const { data } = frt.input()

  const echoEvent = CloudEventCreators.EchoEvent({ status: 'ok', data })

  return frt.output(echoEvent)
})

module.exports = createExports({ handler })
