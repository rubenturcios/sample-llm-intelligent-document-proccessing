import json
from aws_cdk import (
    Environment,
    aws_opensearchserverless as aoss,
    aws_ssm as ssm,
    Stack,
)


class OpensearchCollectionStack(Stack):
    def __init__(
        self, scope: Stack, id: str, *, collection_name: str, iam_access_arns: list[str], env: Environment
    ) -> None:
        super().__init__(scope, id, env=env, cross_region_references=True)
        self.aoss_encyption_policy = aoss.CfnSecurityPolicy(
            self,
            f"{collection_name}EncryptionPolicy",
            name=f"{collection_name}-enc-policy".lower(),
            type="encryption",
            policy=json.dumps(
                {
                    "Rules": [
                        {
                            "ResourceType": "collection",
                            "Resource": [f"collection/{collection_name}"],
                        }
                    ],
                    "AWSOwnedKey": True,
                }
            ),
        )
        self.aoss_network_policy = aoss.CfnSecurityPolicy(
            self,
            f"{collection_name}NetworkPolicy",
            name=f"{collection_name}-net-policy",
            type="network",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                            },
                            {
                                "ResourceType": "dashboard",
                                "Resource": [f"collection/{collection_name}"],
                            },
                        ],
                        "AllowFromPublic": True,
                    }
                ]
            ),
        )
        self.aoss_data_access_policy = aoss.CfnAccessPolicy(
            self,
            f"{collection_name}AccessPolicy",
            name=f"{collection_name}-access-policy",
            policy=json.dumps(
                [
                    {
                        "Rules": [
                            {
                                "ResourceType": "index",
                                "Resource": [f"index/{collection_name}/*"],
                                "Permission": [
                                    "aoss:UpdateIndex",
                                    "aoss:DescribeIndex",
                                    "aoss:ReadDocument",
                                    "aoss:WriteDocument",
                                    "aoss:CreateIndex",
                                    "aoss:DeleteIndex"
                                ],
                            },
                            {
                                "ResourceType": "collection",
                                "Resource": [f"collection/{collection_name}"],
                                "Permission": [
                                    "aoss:DescribeCollectionItems",
                                    "aoss:CreateCollectionItems",
                                    "aoss:UpdateCollectionItems",
                                ],
                            },
                        ],
                        "Principal": iam_access_arns,
                    }
                ]
            ),
            type="data",
        )

        self.opensearch_collection = aoss.CfnCollection(
            self,
            collection_name,
            name=collection_name,
            type="VECTORSEARCH",
        )
        self.opensearch_collection.add_dependency(self.aoss_network_policy)
        self.opensearch_collection.add_dependency(self.aoss_encyption_policy)
        self.opensearch_collection.add_dependency(self.aoss_data_access_policy)
