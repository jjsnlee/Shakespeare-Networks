var termiteTopics = angular.module('termiteTopicDisplay', [], function($locationProvider) {
  $locationProvider.html5Mode(true);
  //angular.bootstrap(document.documentElement);
});

termiteTopics.directive('myDraggable', function($document) {
  return function(scope, element, attr) {
    addDnD($document, scope, element, attr);
  };
});

termiteTopics.service('termiteMsgService', function() {
  var topicHandler = null;
  setTopic = function(topicIndex, topicLabel) {
		topicHandler(topicIndex, topicLabel);
  };
  registerTopicHandler = function(handler) {
  	topicHandler = handler;
  }
  return {
  	registerTopicHandler : registerTopicHandler,
  	setTopic : setTopic
  };
});

termiteTopics.controller('contentCtrl', function($scope, $http, termiteMsgService) {
	$scope.topDocsForTopic = [];
	$scope.selectedTopic = null;

	$scope.getSelected = function(topicIndex, topicLabel) {
		console.log('Got msg about topicIndex: ' + topicIndex);

		$scope.selectedTopic = topicLabel;
		$http.get('/shakespeare/corpus/ldatopics/'+topicIndex).success(function(data) {
			data = data.map(function(c) {
    		return {
    			'char'  : c[0], 
    			'score' : c[1].toFixed(6), 
    			'url'   : ''
    		}
  		});
			$scope.topDocsForTopic = _.sortBy(data, function(c) { 
				return -1*c.score; 
			});
    });
  };
  termiteMsgService.registerTopicHandler($scope.getSelected);
  
  //$scope.getSelected(1);
	/*$scope.setStateModel = function(stateModel, termTopicMatrixView) {
		stateModel.listenTo( termTopicMatrixView, "click:topic", function(topicIndex) {
			console.log('topicIndex' + topicIndex);
		} );  	
  };*/

});
