from googleapiclient import discovery

class PerspectiveClient:
    def __init__(self, api_key):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False
        )

    def analyze_text(self, text, attributes):
        analyze_request = {
            'comment': {'text': text},
            'requested_attributes': {attr: {} for attr in attributes}
        }

        truncated = False
        try: 
            response = self.client.comments().analyze(body=analyze_request).execute()
        except:
            truncated = True

        scores = {}
        if truncated:
            for attr in attributes:
                scores[attr] = 1
        else:
            for attr in attributes:
                scores[attr] = response['attributeScores'][attr]['summaryScore']['value']

        return scores, truncated