import React from 'react';
import { render } from 'react-dom';

export default class SpeakerDeck extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return this.props.dataId ? (
      <div className="uk-section">
        <h2 className="uk-text-bold uk-heading-line">
          <span>Slides</span>
        </h2>
        <div
          style={{
            left: 0,
            width: '100%',
            height: 0,
            position: 'relative',
            paddingBottom: '56.1987%',
            marginBottom: '2rem',
          }}
        >
          <iframe
            src={`//speakerdeck.com/player/${this.props.dataId}`}
            style={{
              border: 0,
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              position: 'absolute',
            }}
            frameBorder="0"
            allowFullScreen
          ></iframe>
        </div>
      </div>
    ) : null;
  }
}
