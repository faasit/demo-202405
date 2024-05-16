const CloudEventTypes = {
  EchoEvent: 'example.echo',
}

const CloudEventCreators = Object.fromEntries(
  Object.keys(CloudEventTypes).map((k) => [k, (d) => d])
);

module.exports = { CloudEventTypes, CloudEventCreators }
