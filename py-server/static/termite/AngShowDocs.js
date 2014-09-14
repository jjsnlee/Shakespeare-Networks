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
    termFrequencyModel = m['termFrequencyModel'];
    stateModel = m['stateModel'];
  }

  var models, termTopicProbModel, termFrequencyModel, stateModel;
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
    $http.get('/shakespeare/corpus/lda/'+LdaModel+'/'+topicIndex).success(function(data) {
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
  
  // probably a much better way to do this...
	if(stateModel.get('doubleClickTopic')) {
	  var topicIndex = stateModel.get('doubleClickTopic');
	  //var topicLabel = termTopicMatrixView.parentModel.attributes.topicIndex[topicIndex];
	  //var topicLabel = termTopicProbModel.attributes.topicIndex[topicIndex];
	  var topicLabel = 'Topic '+(topicIndex+1);
	  getSelectedTopic($scope.$parent.LDAModel, topicIndex, topicLabel);
	}
});
