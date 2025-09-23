// Dynamic Simulation Engine for Peace Simulator
// =============================================

// AI Agent Thoughts Generator for Dev Mode
function generateAIThoughts(year, month, playerRole, lastDecision) {
    const thoughts = {
        1930: {
            "Tosei-ha": [
                "經濟危機提供了擴張的機會，但需要謹慎規劃...",
                "應該先在滿洲建立基礎，而非立即全面開戰...",
                "控制派認為需要先整合國內資源再考慮擴張..."
            ],
            "Kwantung Army": [
                "石原莞爾正在制定滿洲事變計劃，時機尚未成熟...",
                "需要製造合適的藉口，不能過早行動...",
                "張學良的東北軍實力不容小覷，需要周密計劃..."
            ],
            "Zaibatsu": [
                "經濟蕭條影響出口，需要新市場但戰爭風險太大...",
                "投資滿洲鐵路可能比軍事冒險更有利可圖...",
                "與美國貿易關係惡化將是災難性的..."
            ],
            "Chinese Communists": [
                "紅軍正在江西建立根據地，暫時無力對抗日本...",
                "應該利用民族危機壯大自己的力量...",
                "國民黨圍剿是當前主要威脅，日本是次要矛盾..."
            ],
            "Chiang KMT": [
                "必須先安內後攘外，共產黨是心腹大患...",
                "日本野心明顯，但現在軍力不足以對抗...",
                "需要爭取時間進行軍事現代化和經濟建設..."
            ]
        },
        1931: {
            "Tosei-ha": [
                "三月事件失敗顯示需要更謹慎的策略...",
                "關東軍可能自主行動，需要加強控制...",
                "九一八的時機似乎已經成熟..."
            ],
            "Kwantung Army": [
                "柳條湖計劃已經準備就緒，等待最佳時機...",
                "東京的猶豫不決令人沮喪，必須自主行動...",
                "製造事件比等待政府批准更有效..."
            ]
        },
        1936: {
            "Tosei-ha": [
                "二二六事件後我們完全控制了軍部...",
                "對華全面戰爭的準備已經就緒...",
                "西安事變改變了中國局勢，需要重新評估..."
            ]
        },
        1937: {
            "Tosei-ha": [
                "盧溝橋的緊張局勢可以利用為開戰藉口...",
                "中國的抵抗意志似乎被低估了...",
                "需要速戰速決，避免長期消耗戰..."
            ]
        }
    };

    const yearThoughts = thoughts[year] || thoughts[1930];
    const agents = Object.keys(yearThoughts);
    const result = [];

    agents.forEach(agent => {
        const agentThoughts = yearThoughts[agent];
        const thought = agentThoughts[Math.floor(Math.random() * agentThoughts.length)];
        result.push({ agent, thought });
    });

    return result;
}

// Random Incident Generator
function generateRandomIncident(year, month, warIntention) {
    const incidents = [
        // Low tension incidents (warIntention < 30)
        {
            minTension: 0, maxTension: 30,
            events: [
                {
                    title: "邊境走私事件",
                    description: "滿洲邊境發現大規模走私，雙方指責對方失職。",
                    peaceEffect: -2,
                    warEffect: 3
                },
                {
                    title: "文化交流成功",
                    description: "中日學生交流活動圓滿結束，增進相互理解。",
                    peaceEffect: 5,
                    warEffect: -3
                },
                {
                    title: "貿易談判進展",
                    description: "雙方就關稅問題達成初步共識。",
                    peaceEffect: 3,
                    warEffect: -2
                }
            ]
        },
        // Medium tension incidents (warIntention 30-60)
        {
            minTension: 30, maxTension: 60,
            events: [
                {
                    title: "間諜事件",
                    description: "日本間諜在華被捕，外交關係緊張。",
                    peaceEffect: -5,
                    warEffect: 8
                },
                {
                    title: "士兵衝突",
                    description: "邊境巡邏隊發生小規模交火，無人傷亡。",
                    peaceEffect: -8,
                    warEffect: 10
                },
                {
                    title: "僑民騷亂",
                    description: "日本僑民與當地居民發生衝突，軍方要求保護。",
                    peaceEffect: -6,
                    warEffect: 7
                }
            ]
        },
        // High tension incidents (warIntention 60-90)
        {
            minTension: 60, maxTension: 90,
            events: [
                {
                    title: "軍官擅自行動",
                    description: "前線軍官未經授權發動攻擊，局勢升級！",
                    peaceEffect: -15,
                    warEffect: 20
                },
                {
                    title: "暗殺未遂",
                    description: "重要官員遭暗殺未遂，雙方互相指責。",
                    peaceEffect: -12,
                    warEffect: 15
                },
                {
                    title: "軍艦對峙",
                    description: "海軍艦隊在公海對峙，一觸即發。",
                    peaceEffect: -10,
                    warEffect: 12
                }
            ]
        }
    ];

    // Find appropriate incident category
    const category = incidents.find(cat =>
        warIntention >= cat.minTension && warIntention < cat.maxTension
    ) || incidents[0];

    // 20% chance of incident per turn
    if (Math.random() > 0.8) {
        const incident = category.events[Math.floor(Math.random() * category.events.length)];
        return incident;
    }

    return null;
}

// Historical Context Generator
function getDetailedHistoricalContext(year, month) {
    const contexts = {
        1930: {
            1: "倫敦海軍會議召開，日本面臨限制海軍的壓力。經濟大蕭條深化。",
            3: "三月事件醞釀中，軍部激進派計劃政變。",
            6: "農村經濟崩潰，軍部利用民怨推動擴張主義。",
            9: "東北軍閥張學良鞏固對滿洲的控制。",
            12: "關東軍開始制定來年的滿洲計劃。"
        },
        1931: {
            3: "三月事件失敗，但軍部影響力持續增長。",
            6: "中村大尉事件，日軍間諜在滿洲被殺。",
            9: "柳條湖事件（九一八）可能發生，關東軍蠢蠢欲動。",
            10: "如九一八發生，國際聯盟開始關注。",
            12: "年末局勢取決於之前的決策。"
        },
        1932: {
            1: "一二八事變可能在上海發生。",
            3: "滿洲國可能成立（如果九一八已發生）。",
            5: "五一五事件，首相犬養毅可能被刺。",
            10: "國聯李頓調查團活動。"
        },
        1933: {
            2: "日本可能退出國際聯盟。",
            3: "長城抗戰可能發生。",
            5: "塘沽協定談判時期。"
        },
        1934: {
            all: "相對平靜期，各方整備力量。日本確立大東亞共榮圈概念。"
        },
        1935: {
            all: "華北事變風險期。日本試圖分離華北。"
        },
        1936: {
            2: "二二六事件，軍部政變企圖。",
            12: "西安事變，改變中國政治格局。"
        },
        1937: {
            7: "盧溝橋事變風險極高。",
            8: "如戰爭爆發，淞滬會戰可能開始。",
            12: "南京危機（如戰爭已全面爆發）。"
        },
        1938: { all: "戰爭擴大或和平鞏固的關鍵年。" },
        1939: { all: "歐洲局勢影響亞洲，二戰可能全面爆發。" },
        1940: { all: "最終考驗 - 是否維持了和平？" }
    };

    const yearContext = contexts[year] || contexts[1930];
    return yearContext[month] || yearContext.all || `${year}年${month}月：局勢發展中...`;
}

// Dynamic News Generator
function generateDynamicNews(year, month, warIntention, lastActions) {
    const newsTemplates = {
        economic: [
            `${year}年經濟報告：失業率${15 + Math.random() * 10}%，民眾不滿增加`,
            `財閥要求政府保護海外投資，施壓${warIntention > 50 ? '軍事' : '外交'}解決`,
            `農產品價格暴跌${20 + Math.random() * 30}%，農村動盪加劇`,
            `新工業計劃：${warIntention > 60 ? '軍工' : '民用'}產業投資增加`
        ],
        military: [
            `陸軍申請預算增加${10 + warIntention/5}%，${warIntention > 70 ? '獲批准' : '遭質疑'}`,
            `${warIntention > 50 ? '鷹派' : '鴿派'}將領在參謀本部影響力上升`,
            `新兵訓練強度${warIntention > 60 ? '顯著提高' : '維持正常'}`,
            `軍方情報：${Math.random() > 0.5 ? '誇大' : '低估'}了鄰國軍力`
        ],
        diplomatic: [
            `${warIntention < 40 ? '和平' : '強硬'}派外交官提出新${warIntention < 50 ? '合作' : '施壓'}方案`,
            `國際輿論${warIntention > 60 ? '擔憂' : '讚賞'}最近的外交動向`,
            `第三國表示願意${warIntention < 50 ? '調停' : '觀望'}`,
            `外交部內部${warIntention > 70 ? '邊緣化' : '影響力恢復'}`
        ],
        social: [
            `民間${warIntention > 60 ? '戰爭' : '和平'}情緒高漲，遊行人數過萬`,
            `知識分子發表${warIntention < 40 ? '和平' : '愛國'}宣言`,
            `宗教團體呼籲${warIntention < 50 ? '和平對話' : '保衛國家'}`,
            `青年組織活動${warIntention > 70 ? '激進化' : '理性化'}`
        ]
    };

    const categories = Object.keys(newsTemplates);
    const selectedCategory = categories[Math.floor(Math.random() * categories.length)];
    const newsItems = newsTemplates[selectedCategory];
    const selectedNews = newsItems[Math.floor(Math.random() * newsItems.length)];

    return selectedNews;
}

// Export functions for use in main HTML
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        generateAIThoughts,
        generateRandomIncident,
        getDetailedHistoricalContext,
        generateDynamicNews
    };
}