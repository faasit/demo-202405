export declare var CloudEventTypes: {
  EchoEvent: 'example.echo'
}

export declare var CloudEventCreators: {
  EchoEvent: (d: EchoEvent) => EchoEvent
}

export interface EchoEvent {
  status: string
  data: any
}

