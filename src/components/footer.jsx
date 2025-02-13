import React from 'react';
import { render } from 'react-dom';
import { FaCreativeCommons } from 'react-icons/fa';

export default class Footer extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-text-center uk-text-small">
        <p>
          powered by <FaCreativeCommons />{' '}
          <a
            href="https://github.com/denkiwakame/academic-project-template"
            target="_blank"
          >
            {' '}
            Academic Project Page Template{' '}
          </a>
        </p>
      </div>
    );
  }
}
