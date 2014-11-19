var termiteTopics = angular.module('termiteTopicDisplay', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
  //angular.bootstrap(document.documentElement);
});

termiteTopics.directive('myDraggable', function($document) {
  return function(scope, element, attr) {
    addDnD($document, scope, element, attr);
  };
});

//termiteTopics.factory('stateModel', function() {
//  return new StateModel();
//});

termiteTopics.service('termiteMsgService', function() {
  var topicHandler = null;
  var setTopic = function(LdaModel, topicIndex, topicLabel) {
		topicHandler(LdaModel, topicIndex, topicLabel);
  };
  var registerTopicHandler = function(handler) { topicHandler = handler; };
  //var registerModelHandler = function(handler) { topicHandler = handler; };

  return {
  	registerTopicHandler : registerTopicHandler,
  	setTopic : setTopic,
  };
});

termiteTopics.controller('msgCtrl', function($scope, $http, $window, termiteMsgService) {

  var stateModel, parentTTPModel, ttProbModel, tFreqModel;  

  function loadPage(ldaModel) {
    // create backbone models and views
    stateModel = new StateModel();
    var ttProbView = new TermTopicMatrixView( {el:"div.termTopicMatrixContainer"} );
    var tFreqView = new TermFrequencyView( {el:"div.termFrequencyContainer"} );
  
    // Almost absolutely nothing here atm...  
    parentTTPModel = new ParentTermTopicProbabilityModel();
    // Model for the termTopicMatrixView
    ttProbModel = new TermTopicProbabilityModel();
    // Model for the term frequency column
    tFreqModel = new TermFrequencyModel();  

      // init user control views
    var totalTermsView = new TotalTermsView( {model: stateModel} );
    var affinityNumTermsView = new AffinityNumTermsView( {model: stateModel} );
    var salientNumTermsView = new SalientNumTermsView( {model: stateModel} );

    initUserControlViewComponents(stateModel);

    var basedir = '/shakespeare/corpus/topicModels/'+ldaModel+'/termite'
    parentTTPModel.url = basedir+"/seriated-parameters.json";
    ttProbModel.url = basedir+"/filtered-parameters.json";
    tFreqModel.url = basedir+"/global-term-freqs.json";

    // Pretty msure don't need this parentTTPModel, the tFreqModel
    // should pretty much have the same data...

    ttProbModel.initModel( parentTTPModel, stateModel );
    //ttProbModel.initModel( tFreqModel, stateModel );
    ttProbView.initModel( ttProbModel );

    tFreqModel.initModels( ttProbModel, stateModel );
    tFreqView.initModel( tFreqModel );

    // load states from QueryString and save changes to state to query string
    stateModel.on( "change", stateModel.saveStatesToQueryString, stateModel );
    
    // load all models
    stateModel.once( "loaded:states", parentTTPModel.load, parentTTPModel );
    //stateModel.once( "loaded:states", ttProbModel.load, ttProbModel );
    parentTTPModel.once( "loaded:seriated", ttProbModel.load, ttProbModel );
    ttProbModel.once( "loaded:filtered", ttProbView.load, ttProbView );
    ttProbModel.once( "loaded:filtered", tFreqModel.load, tFreqModel );
    tFreqModel.once( "loaded:freqModel", tFreqView.load, tFreqView );

    stateModel.once("sending:colors", ttProbView.receiveSelectedTopics, ttProbView);

    // initialize all events that listen to stateModel
    stateModel.once( "loaded:states", function() {
      // can probably dump all of these 
      ttProbModel.listenTo(stateModel, "change:numAffinityTerms", ttProbModel.update.bind( ttProbModel ));
      ttProbModel.listenTo(stateModel, "change:numSalientTerms", ttProbModel.update.bind( ttProbModel ));
      ttProbModel.listenTo(stateModel, "change:addTopTwenty", ttProbModel.update.bind( ttProbModel ));
      
      ttProbView.listenTo(stateModel, "change:highlightedTerm", ttProbView.onSelectionTermChanged, ttProbView );
      ttProbView.listenTo(stateModel, "change:highlightedTopic", ttProbView.onSelectionTopicChanged, ttProbView );
      
      tFreqView.listenTo(stateModel, "change:highlightedTerm", tFreqView.onHighlightTermChanged, tFreqView);
      tFreqView.listenTo(stateModel, "change:highlightedTopic", tFreqView.onHighlightTopicChanged, tFreqView);
      
      ttProbView.listenTo( stateModel, "color:topic", ttProbView.clickTopic, ttProbView);
      ttProbModel.listenTo( stateModel, "color:topic", ttProbModel.selectTopic, ttProbModel);
      tFreqModel.listenTo( stateModel, "color:topic", tFreqModel.selectTopic, tFreqModel);
    });

    // initialize all events that listen to filtered model
    ttProbModel.once( "loaded:filtered", function() {
      // Declare dependencies
      // data pipeline events
      tFreqModel.listenTo( ttProbModel, "change:termIndex", tFreqModel.update.bind( tFreqModel ));
      ttProbView.listenTo( ttProbModel, "change:sparseMatrix", ttProbView.update.bind( ttProbView ));
      totalTermsView.listenTo( ttProbModel, 'change:termIndex', totalTermsView.render );
    
      // Once the matrix has loaded, need to set the top level div to a fixed size,
      // and then can set the inner one to be wider, which the scrollbar 
      $(".wrapper1").width($(".wrapper1").width());
      $(".div1").width($(".termTopicMatrixContainer").children().width());
    });

    // initialize all events that listen to term frequency model
    tFreqModel.once( "loaded:freqModel", function() {
      tFreqView.listenTo( tFreqModel, "change:topicalFreqMatrix", tFreqView.renderUpdate, tFreqView);
      tFreqView.listenTo( tFreqModel, "change:termIndex", tFreqView.renderUpdate.bind( tFreqView ));
    });

    // initialize user controls listeners that catch state model changes
    affinityNumTermsView.listenTo( stateModel, 'change:numAffinityTerms', affinityNumTermsView.render );
    salientNumTermsView.listenTo( stateModel, 'change:numSalientTerms', salientNumTermsView.render );
    totalTermsView.listenTo( stateModel, 'change:totalTerms', totalTermsView.render );

    // initialize state model listeners that catch view events
    stateModel.listenTo( ttProbView, "mouseover:topic mouseout:topic", stateModel.setHighlightedTopic );
    stateModel.listenTo( ttProbView, "mouseover:term mouseout:term", stateModel.setHighlightedTerm );
    stateModel.listenTo( tFreqView,  "mouseover:term mouseout:term", stateModel.setHighlightedTerm );

    // http://backbonejs.org/#Events-listenTo: object.listenTo(other, event, callback)
    stateModel.listenTo( ttProbView, "click:topic", function(topicIndex) {
      console.log('[click:topic] topicIndex: '+topicIndex+', highlightedTopic: '+stateModel.get('highlightedTopic'));
      //console.log( ttProbView.parentModel );
      // We could be clicking OFF a topic 
      if(stateModel.get('highlightedTopic') == topicIndex) {
	      stateModel.setDoubleClickTopic(topicIndex);
	      stateModel.selectTopic(topicIndex);
        var topicLabel = ttProbView.parentModel.attributes.topicIndex[topicIndex];
        termiteMsgService.setTopic(ldaModel, topicIndex, topicLabel);
      }
      else {
        // lifted from statemodel.selectTopic
        if(topicIndex in stateModel.get("selectedTopics")) {
	        freeColor(stateModel.get("selectedTopics")[topicIndex]);
	        delete stateModel.get("selectedTopics")[topicIndex];
        }        
      }
    });
    // the loading process is complete?
    stateModel.loadStatesFromQueryString();
    initColorObjects(null);
  }
  
  $scope.getModels = function() {
    return {
      stateModel         : stateModel, 
      termTopicProbModel : ttProbModel,
      tFreqModel : tFreqModel 
    };
  };

  $scope.changeModel = function() {
    console.log('$scope.LDAModel: '+$scope.LDAModel);
    $('.termTopicMatrixContainer').empty();
    $('.termFrequencyContainer').empty();
    loadPage($scope.LDAModel);
    $scope.$$childHead.setModels($scope.getModels());

    $scope.$$childHead.topDocsForTopic = [];
      $scope.$$childHead.docName    = '';
      $scope.$$childHead.docContent = '';
  };

  $http.get('/shakespeare/corpus/topicModels').success(function(data) {
      $scope.LDAModels = data;
      $scope.LDAModel = $scope.LDAModels[0];
      loadPage($scope.LDAModel);
      $scope.$$childHead.setModels($scope.getModels());
    }).error(function(data) {
      $scope.LDAModels = ['Error loading models...'];
      $scope.LDAModel = $scope.LDAModels[0];
    });
});

termiteTopics.controller('contentCtrl', function($scope, $http, $sce, termiteMsgService) {
  $scope.topDocsForTopic = [];
  
  // As a string
  $scope.selectedTopic = null;
  // As an index
  var selectedTopicIndex = -1;
  
  $scope.showTopicDetails = 1;

  $scope.setModels = function(m) {
    models = m;
    termTopicProbModel = m['termTopicProbModel'];
    tFreqModel = m['tFreqModel'];
    stateModel = m['stateModel'];
    // probably a much better way to do this...
    if(stateModel && stateModel.get('doubleClickTopic')) {
      var topicIndex = stateModel.get('doubleClickTopic');
      //var topicLabel = ttProbView.parentModel.attributes.topicIndex[topicIndex];
      //var topicLabel = termTopicProbModel.attributes.topicIndex[topicIndex];
      var topicLabel = 'Topic '+(topicIndex+1);
      getSelectedTopic($scope.$parent.LDAModel, topicIndex, topicLabel);
    }
  }

  var models, termTopicProbModel, tFreqModel, stateModel;
  $scope.setModels($scope.$parent.getModels());
  
  $scope.colorAxis = {
    //colors: ['#', '#','#', '#', '#', '#'],
    //colors: ['#E7E0D9', '#E7CFB7','#E7B98A', '#FFCE9E', '#FFB164', '#FF7F00'],
    colors: ['#A89D7D', '#A89049','#E7AD00', '#FFB164', '#FFFF00', '#DFFF00'],
    values: [0, 1, 10, 25, 50, 100]
  };

  var scoreColor = function(score) {
    var colorIdx = 0;
    for(i in $scope.colorAxis.values) {
      if(score < $scope.colorAxis.values[i]) {
        break;
      }
      colorIdx = i;
    }
    return $scope.colorAxis.colors[colorIdx]; 
  }
  
  $scope.getScore = function(topicIdx, termIdx) {
    var termTopicScoreMatrix = termTopicProbModel.get('matrix');
    var termScore = termTopicScoreMatrix[termIdx][topicIdx];
    console.log('Term score is: ' + termScore);
  }

  $scope.getDocContent = function(charNm) {
    console.log('charNm: '+charNm);
    $scope.showTopicDetails = 1;
    $http.get('/shakespeare/corpus/characters/'+charNm).success(function(data) {
      console.log('data: '+data);
      var termTopicScoreMatrix = termTopicProbModel.get('matrix');
      var terms = termTopicProbModel.get('termIndex');
      var content = data.doc_content.map(function(section) {
        var sectionText = section.map(function(li) {
          // need to fix this for punctuation - ie, Jew? is not getting picked up
          var wds = li.split(/[\s?.,;:]/);
          var uniqueWds = {};
          _.each(wds, function(wd) {
            wd = wd.toLowerCase();
            if(wd in uniqueWds) 
              return;
            uniqueWds[wd] = 1;
            var idx = terms.indexOf(wd);
            if(idx>-1) {
              var termScore = termTopicScoreMatrix[idx][selectedTopicIndex];
              if(termScore) {
	              var heatMapColor = scoreColor(termScore);
	              var getScoreArgs = selectedTopicIndex+','+idx;
	              li = li.replace(new RegExp('\\b('+wd+')\\b', 'gi'), 
	                    "<span ng-click='getScore("+getScoreArgs+")' style='font-weight:bold;background-color:"
	                       +heatMapColor+"'>$1</span>" );
	            }
            }
          });
          return li;
        }).join('<br>\n');
        
        return '<p class="charText">'+sectionText+'</p>'
      }).join('\n');
      
      content = $sce.trustAsHtml(content);
      $scope.docName    = data.doc_name;
      $scope.docContent = content;
    }).error(function(data) {
      var content = $sce.trustAsHtml(data);
      $scope.docContent = content;
    });
    
    // Need a failure
    
  };
  
  function getSelectedTopic(LdaModel, topicIndex, topicLabel) {
  
    if(topicIndex==selectedTopicIndex)
      return;
  
    // First reset the existing 
    $scope.topDocsForTopic = [];
	  $scope.docName    = '';
	  $scope.docContent = '';
  
    console.log('Will fetch data for topicIndex: ' + topicIndex);
    $scope.selectedTopic = topicLabel;
    selectedTopicIndex = topicIndex;
    $http.get('/shakespeare/corpus/topicModels/'+LdaModel+'/'+topicIndex).success(function(data) {
      data = data.map(function(c) {
        return {
          'char'  : c[0], 
          'score' : c[1].toFixed(6)
        }
      });
      $scope.topDocsForTopic = _.sortBy(data, function(c) { 
        return -1*c.score; 
      });
    });
  };
  
  termiteMsgService.registerTopicHandler(getSelectedTopic);
});
