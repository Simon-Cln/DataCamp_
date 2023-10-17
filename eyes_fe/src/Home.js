import React from 'react';
import Presentation from './Presentation';
import Detection from './Detection';
import Verticaline  from './Verticaline';

function Home() {
  return (
    <div>
      <div id="presentation">
        <Presentation />
      </div>
      <Verticaline />
      <div id="detection">
        <Detection />
      </div>
    </div>
  );
}

export default Home;
