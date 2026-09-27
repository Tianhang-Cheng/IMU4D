(function () {
  'use strict';

  var template = document.getElementById('mc-sample-template');
  if (!template) return;

  // Each .mc-root names the window globals holding its manifests (data-manifest, comma
  // separated); samples of all of them are merged, so one block can mix datasets built
  // by different comparison runs. Methods and protocol come from the first manifest.
  Array.prototype.forEach.call(document.querySelectorAll('.mc-root'), function (root) {
    var manifests = (root.dataset.manifest || 'COMPARISON_MANIFEST').split(',').map(function (name) {
      return window[name.trim()];
    });
    if (manifests.some(function (m) { return !m; })) {
      setup(root, null);
      return;
    }
    setup(root, Object.assign({}, manifests[0], {
      samples: manifests.reduce(function (all, m) { return all.concat(m.samples); }, [])
    }));
  });

  function setup(root, data) {
  var loading = root.querySelector('.mc-loading');
  if (!data) {
    loading.textContent = 'Comparison manifest is missing. Run the model comparison build command first.';
    return;
  }

  var samplesRoot = root.querySelector('.mc-samples');
  var staticFallback = root.querySelector('.mc-static-fallback');
  var methods = data.methods;
  var carousels = [];
  // Initial auto-next state comes from the button's aria-pressed in the HTML.
  var autoNextButton = root.querySelector('.mc-autonext');
  var autoNext = !!autoNextButton && autoNextButton.getAttribute('aria-pressed') === 'true';

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return '0:00';
    return Math.floor(seconds / 60) + ':' + String(Math.floor(seconds % 60)).padStart(2, '0');
  }

  // Manifests list one row per text source; older ones map method key -> text.
  function textRows(sample) {
    if (Array.isArray(sample.text_results)) return sample.text_results;
    var results = sample.text_results || { gt: sample.caption || '' };
    return methods.map(function (method) {
      return {
        key: method.key,
        label: method.key === 'gt' ? 'GT'
          : (method.key === 'mobileposer' ? 'MobilePoser → MotionGPT3' : method.display_name),
        color: method.color,
        text: results[method.key] || '',
        note: method.key === 'imuposer' ? 'Motion only (no text prediction)' : ''
      };
    });
  }

  function buildCard(sample) {
    var card = template.content.firstElementChild.cloneNode(true);
    card.dataset.dataset = sample.dataset;
    card.dataset.panels = String((sample.roles || methods).length);

    var textRoot = card.querySelector('.mc-text-results');
    // data-hide-text on the .mc-root drops the text rows under each video.
    if ('hideText' in root.dataset) textRoot.hidden = true;
    // data-text-keys="ours" keeps only those text rows (comma-separated keys).
    var textKeys = root.dataset.textKeys ? root.dataset.textKeys.split(',') : null;
    textRows(sample).filter(function (entry) {
      return !textKeys || textKeys.indexOf(entry.key) !== -1;
    }).forEach(function (entry) {
      var row = document.createElement('div');
      row.className = 'mc-text-result';
      row.style.setProperty('--mc-method-color', entry.color);

      var label = document.createElement('span');
      label.className = 'mc-text-label';
      label.textContent = entry.label;

      var result = document.createElement('span');
      result.className = 'mc-text-value';
      if (entry.text) {
        result.textContent = entry.text;
      } else {
        result.classList.add('is-unavailable');
        result.textContent = entry.key === 'gt' ? '(no annotation)' : '(no text output)';
      }
      row.appendChild(label);
      row.appendChild(result);
      textRoot.appendChild(row);
    });

    var video = card.querySelector('.mc-composite-video');
    video.dataset.src = sample.comparison_video;
    // Posters are fetched only when a card is about to be shown; fetching all of them
    // up front competes with the videos for bandwidth and connections.
    var prepared = false;
    function prepare() {
      if (prepared) return;
      prepared = true;
      video.poster = sample.poster;
      // Size the box to this clip's composite (portrait v3 panels are narrower than
      // the rest), so object-fit never letterboxes it; the CSS ratio is only a fallback.
      var posterImage = new Image();
      posterImage.onload = function () {
        video.style.aspectRatio = posterImage.naturalWidth + ' / ' + posterImage.naturalHeight;
      };
      posterImage.src = sample.poster;
    }
    var play = card.querySelector('.mc-play');
    var timeline = card.querySelector('.mc-timeline');
    var clock = card.querySelector('.mc-clock');
    var speed = card.querySelector('.mc-speed');
    var playing = false;

    function ensureLoaded() {
      if (!video.src) {
        video.src = video.dataset.src;
        video.load();
      }
    }
    function start() {
      ensureLoaded();
      video.playbackRate = Number(speed.value);
      video.play().catch(function () {});
      playing = true;
      play.textContent = 'Pause';
    }
    function pause() {
      video.pause();
      playing = false;
      play.textContent = 'Play';
    }
    // Fetch ahead so auto-next can start without waiting on the network.
    function preload() {
      prepare();
      video.preload = 'auto';
      ensureLoaded();
    }
    // Drop the source of a card that is off screen. A paused, half-downloaded video keeps
    // its HTTP connection open, and enough of them starve new videos of connections.
    function unload() {
      if (!video.src) return;
      pause();
      video.removeAttribute('src');
      video.load();
      video.preload = 'none';
      timeline.value = 0;
      clock.textContent = '0:00 / 0:00';
    }

    var record = { card: card, video: video, userPaused: false, start: start, pause: pause, prepare: prepare, preload: preload, unload: unload, onEnded: null };

    play.addEventListener('click', function () {
      if (playing) {
        pause();
        record.userPaused = true;
      } else {
        record.userPaused = false;
        start();
      }
    });
    timeline.addEventListener('input', function () {
      ensureLoaded();
      var duration = video.duration || sample.duration_seconds;
      var time = Number(timeline.value) / 1000 * duration;
      video.currentTime = time;
      clock.textContent = formatTime(time) + ' / ' + formatTime(duration);
    });
    speed.addEventListener('change', function () {
      video.playbackRate = Number(speed.value);
    });
    video.addEventListener('timeupdate', function () {
      if (!video.duration) return;
      timeline.value = Math.round(video.currentTime / video.duration * 1000);
      clock.textContent = formatTime(video.currentTime) + ' / ' + formatTime(video.duration);
    });
    video.addEventListener('ended', function () {
      if (playing && record.onEnded) record.onEnded();
    });

    return record;
  }

  function buildCarousel(dataset, datasetName, samples) {
    var section = document.createElement('section');
    section.className = 'mc-carousel';
    section.innerHTML =
      '<div class="mc-carousel-bar">' +
        '<h4 class="mc-carousel-title"></h4>' +
      '</div>' +
      '<div class="mc-carousel-track"></div>' +
      '<div class="mc-carousel-nav">' +
        '<div class="mc-dots"></div>' +
        '<span class="mc-count"></span>' +
      '</div>';
    section.querySelector('.mc-carousel-title').textContent = datasetName;
    // data-hide-group-title on the .mc-root drops the dataset name above each carousel.
    if ('hideGroupTitle' in root.dataset) section.querySelector('.mc-carousel-bar').hidden = true;
    var dotsRoot = section.querySelector('.mc-dots');
    var count = section.querySelector('.mc-count');
    var track = section.querySelector('.mc-carousel-track');

    var carousel = { section: section, records: [], index: -1, visible: false };
    var dots = samples.map(function (sample, i) {
      var record = buildCard(sample);
      record.card.hidden = true;
      record.card.querySelector('.mc-nav-prev').addEventListener('click', function () { carousel.show(carousel.index - 1); });
      record.card.querySelector('.mc-nav-next').addEventListener('click', function () { carousel.show(carousel.index + 1); });
      // At the end of a clip, advance to the next sample, or replay this one when auto-next is off.
      record.onEnded = function () {
        if (autoNext && samples.length > 1) {
          carousel.show(carousel.index + 1);
        } else {
          record.card.querySelector('.mc-composite-video').currentTime = 0;
          record.start();
        }
      };
      track.appendChild(record.card);
      carousel.records.push(record);

      var dot = document.createElement('button');
      dot.type = 'button';
      dot.className = 'mc-dot';
      dot.setAttribute('aria-label', 'Sample ' + (i + 1));
      dot.title = 'Sample ' + (i + 1);
      dot.addEventListener('click', function () { carousel.show(i); });
      // Hovering a dot starts the download, so the clip is usually buffered by the click.
      dot.addEventListener('pointerenter', function () { record.preload(); });
      dot.addEventListener('focus', function () { record.preload(); });
      dotsRoot.appendChild(dot);
      return dot;
    });

    carousel.current = function () { return carousel.records[carousel.index]; };
    carousel.autoplay = function () {
      var record = carousel.current();
      if (record && carousel.visible && !record.userPaused) record.start();
    };
    carousel.show = function (i) {
      var n = carousel.records.length;
      i = (i + n) % n;
      if (i === carousel.index) return;
      if (carousel.index >= 0) {
        carousel.current().pause();
        carousel.current().card.hidden = true;
      }
      carousel.index = i;
      var record = carousel.current();
      var next = carousel.records[(i + 1) % n];
      var prev = carousel.records[(i - 1 + n) % n];
      carousel.records.forEach(function (other) {
        if (other !== record && other !== next && other !== prev) other.unload();
      });
      record.prepare();
      record.card.hidden = false;
      count.textContent = (i + 1) + ' / ' + n;
      dots.forEach(function (dot, k) { dot.classList.toggle('is-active', k === i); });
      carousel.autoplay();
      // Start fetching the neighbouring samples once this one can play through. A card that was
      // itself preloaded has already fired canplaythrough, so check readyState too;
      // otherwise every other transition would start from a cold load.
      function preloadNeighbours() {
        if (next !== record) next.preload();
        if (prev !== record && prev !== next) prev.preload();
      }
      if (record.video.readyState >= 4) {
        preloadNeighbours();
      } else {
        record.video.addEventListener('canplaythrough', function () {
          if (carousel.current() === record) preloadNeighbours();
        }, { once: true });
      }
    };

    carousel.show(0);
    return carousel;
  }

  // data-hide-datasets="a,b" on the .mc-root drops those datasets from the page;
  // data-datasets="a,b" keeps only those, in that order; data-dataset-names='{"a": "A"}'
  // overrides the display names the manifests carry.
  function list(value) {
    return (value || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
  }
  var hidden = list(root.dataset.hideDatasets);
  var only = list(root.dataset.datasets);
  var names = root.dataset.datasetNames ? JSON.parse(root.dataset.datasetNames) : {};
  var groups = [];
  data.samples.forEach(function (sample) {
    if (hidden.indexOf(sample.dataset) !== -1) return;
    if (only.length && only.indexOf(sample.dataset) === -1) return;
    if (names[sample.dataset]) {
      // The text-row notes name the dataset too ("NCSA v3 has no text annotations").
      var oldName = sample.dataset_name, newName = names[sample.dataset];
      var noteName = newName.replace(/\s*\(.*\)$/, '');  // drop the "(3pt, ...)" suffix in notes
      sample = Object.assign({}, sample, {
        dataset_name: newName,
        text_results: (sample.text_results || []).map(function (row) {
          return Object.assign({}, row, { note: (row.note || '').split(oldName).join(noteName) });
        })
      });
    }
    var group = groups.find(function (g) { return g.dataset === sample.dataset; });
    if (!group) {
      group = { dataset: sample.dataset, name: sample.dataset_name, samples: [] };
      groups.push(group);
    }
    group.samples.push(sample);
  });
  if (only.length) {
    groups.sort(function (a, b) { return only.indexOf(a.dataset) - only.indexOf(b.dataset); });
  }
  // data-sample-picks='{"a": [3, 5]}' keeps only those samples (1-based, as the counter shows them).
  var picks = root.dataset.samplePicks ? JSON.parse(root.dataset.samplePicks) : {};
  groups.forEach(function (group) {
    if (picks[group.dataset]) {
      group.samples = picks[group.dataset].map(function (n) { return group.samples[n - 1]; }).filter(Boolean);
    }
  });
  groups.forEach(function (group) {
    var carousel = buildCarousel(group.dataset, group.name, group.samples);
    carousels.push(carousel);
    samplesRoot.appendChild(carousel.section);
  });

  loading.hidden = true;
  if (staticFallback) staticFallback.hidden = true;

  autoNextButton.addEventListener('click', function () {
    autoNext = !autoNext;
    autoNextButton.setAttribute('aria-pressed', String(autoNext));
    autoNextButton.textContent = 'Auto-next: ' + (autoNext ? 'On' : 'Off');
  });

  // Autoplay whichever sample is on screen; pause it once scrolled away.
  if ('IntersectionObserver' in window) {
    var bySection = new Map(carousels.map(function (carousel) { return [carousel.section, carousel]; }));
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var carousel = bySection.get(entry.target);
        carousel.visible = entry.isIntersecting;
        if (carousel.visible) carousel.autoplay();
        else carousel.current().pause();
      });
    }, { threshold: 0.25 });
    carousels.forEach(function (carousel) { observer.observe(carousel.section); });
  } else {
    carousels.forEach(function (carousel) {
      carousel.visible = true;
      carousel.autoplay();
    });
  }
  }
})();
